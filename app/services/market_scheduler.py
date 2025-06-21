"""
Automated Market Hours Scheduling Service

This module provides intelligent scheduling system that automatically updates option analytics
during market hours with timezone handling, holiday calendar integration, different update
frequencies, monitoring and alerting for failed updates and performance metrics.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import asyncio
import pytz
from celery import Celery
from celery.schedules import crontab
import json
import time as time_module

# Internal imports
from app.market_data.market_hours import MarketHoursManager, MarketStatus
from app.analysis.data_pipeline import DataProcessingPipeline, ProcessingMode, ProcessingResult
from app.analysis.option_analytics import OptionAnalyticsEngine
from app.tasks.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)

class ScheduleFrequency(Enum):
    """Scheduling frequency options"""
    EVERY_MINUTE = "every_minute"
    EVERY_5_MINUTES = "every_5_minutes"
    EVERY_15_MINUTES = "every_15_minutes"
    EVERY_30_MINUTES = "every_30_minutes"
    EVERY_HOUR = "every_hour"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduledTask:
    """Configuration for a scheduled task"""
    name: str
    function: str
    frequency: ScheduleFrequency
    priority: TaskPriority
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    exchanges: List[str] = field(default_factory=lambda: ["NSE", "BSE"])
    timezone: str = "Asia/Kolkata"
    retry_count: int = 3
    timeout_seconds: int = 300
    max_runtime_minutes: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduleMetrics:
    """Metrics for scheduled task execution"""
    task_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_symbols_processed: int = 0
    errors: List[str] = field(default_factory=list)

@dataclass
class MarketScheduleState:
    """Current state of market scheduling"""
    is_active: bool = False
    current_market_status: Optional[MarketStatus] = None
    next_market_open: Optional[datetime] = None
    next_market_close: Optional[datetime] = None
    active_tasks: List[str] = field(default_factory=list)
    paused_tasks: List[str] = field(default_factory=list)
    last_status_check: Optional[datetime] = None

class MarketHoursScheduler:
    """
    Intelligent scheduling system for automated market hours-based task execution
    """
    
    def __init__(self):
        self.market_hours = MarketHoursManager()
        self.pipeline = DataProcessingPipeline(mode=ProcessingMode.HYBRID)
        self.analytics_engine = OptionAnalyticsEngine()
        
        # Timezone setup
        self.ist = pytz.timezone('Asia/Kolkata')
        self.utc = pytz.UTC
        
        # Scheduling state
        self.schedule_state = MarketScheduleState()
        self.task_metrics: Dict[str, ScheduleMetrics] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.alert_thresholds = {
            'max_consecutive_failures': 3,
            'max_failure_rate': 0.3,
            'max_average_duration_ms': 60000,  # 1 minute
            'min_success_rate': 0.8
        }
        
        logger.info("Market Hours Scheduler initialized")
        self._initialize_default_schedules()
    
    def _initialize_default_schedules(self):
        """Initialize default scheduling configurations"""
        default_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        
        # High-frequency option chain updates during market hours
        self.add_scheduled_task(ScheduledTask(
            name="option_chain_updates",
            function="update_option_chains",
            frequency=ScheduleFrequency.EVERY_5_MINUTES,
            priority=TaskPriority.HIGH,
            symbols=default_symbols,
            timeout_seconds=120,
            metadata={
                'description': 'Regular option chain data updates',
                'market_hours_only': True
            }
        ))
        
        # Analytics calculations - less frequent but comprehensive
        self.add_scheduled_task(ScheduledTask(
            name="analytics_calculation",
            function="calculate_analytics",
            frequency=ScheduleFrequency.EVERY_15_MINUTES,
            priority=TaskPriority.NORMAL,
            symbols=default_symbols,
            timeout_seconds=300,
            metadata={
                'description': 'Comprehensive analytics calculations',
                'market_hours_only': True,
                'include_historical': True
            }
        ))
        
        # Market open initialization
        self.add_scheduled_task(ScheduledTask(
            name="market_open_init",
            function="market_open_initialization",
            frequency=ScheduleFrequency.MARKET_OPEN,
            priority=TaskPriority.CRITICAL,
            symbols=default_symbols,
            timeout_seconds=600,
            metadata={
                'description': 'Market opening initialization tasks',
                'run_once_per_day': True
            }
        ))
        
        # Market close summary
        self.add_scheduled_task(ScheduledTask(
            name="market_close_summary",
            function="market_close_summary",
            frequency=ScheduleFrequency.MARKET_CLOSE,
            priority=TaskPriority.HIGH,
            symbols=default_symbols,
            timeout_seconds=600,
            metadata={
                'description': 'End of day summary and cleanup',
                'run_once_per_day': True
            }
        ))
        
        # Performance monitoring - runs every hour
        self.add_scheduled_task(ScheduledTask(
            name="performance_monitoring",
            function="monitor_performance",
            frequency=ScheduleFrequency.EVERY_HOUR,
            priority=TaskPriority.LOW,
            symbols=[],  # No symbols needed
            timeout_seconds=60,
            metadata={
                'description': 'System performance monitoring',
                'market_hours_only': False
            }
        ))
    
    def add_scheduled_task(self, task: ScheduledTask):
        """Add a new scheduled task"""
        self.scheduled_tasks[task.name] = task
        self.task_metrics[task.name] = ScheduleMetrics(task_name=task.name)
        logger.info(f"Added scheduled task: {task.name} ({task.frequency.value})")
    
    def remove_scheduled_task(self, task_name: str):
        """Remove a scheduled task"""
        if task_name in self.scheduled_tasks:
            del self.scheduled_tasks[task_name]
            if task_name in self.task_metrics:
                del self.task_metrics[task_name]
            logger.info(f"Removed scheduled task: {task_name}")
    
    def update_market_status(self):
        """Update current market status and scheduling state"""
        try:
            current_status = self.market_hours.get_current_status("NSE")
            
            self.schedule_state.current_market_status = current_status.status
            self.schedule_state.next_market_open = current_status.next_open
            self.schedule_state.next_market_close = current_status.next_close
            self.schedule_state.last_status_check = datetime.now(self.ist)
            
            # Update active/paused tasks based on market status
            self._update_task_states(current_status)
            
            logger.debug(f"Market status updated: {current_status.status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update market status: {str(e)}")
    
    def _update_task_states(self, market_status):
        """Update task states based on current market status"""
        market_hours_only_tasks = [
            name for name, task in self.scheduled_tasks.items()
            if task.metadata.get('market_hours_only', False)
        ]
        
        if market_status.status in [MarketStatus.OPEN, MarketStatus.PRE_OPEN]:
            # Activate market hours tasks
            for task_name in market_hours_only_tasks:
                if task_name not in self.schedule_state.active_tasks:
                    self.schedule_state.active_tasks.append(task_name)
                if task_name in self.schedule_state.paused_tasks:
                    self.schedule_state.paused_tasks.remove(task_name)
        else:
            # Pause market hours tasks
            for task_name in market_hours_only_tasks:
                if task_name in self.schedule_state.active_tasks:
                    self.schedule_state.active_tasks.remove(task_name)
                if task_name not in self.schedule_state.paused_tasks:
                    self.schedule_state.paused_tasks.append(task_name)
    
    def should_execute_task(self, task_name: str) -> bool:
        """Determine if a task should be executed now"""
        if task_name not in self.scheduled_tasks:
            return False
        
        task = self.scheduled_tasks[task_name]
        
        # Check if task is enabled
        if not task.enabled:
            return False
        
        # Check if task requires market hours
        if task.metadata.get('market_hours_only', False):
            if task_name in self.schedule_state.paused_tasks:
                return False
        
        # Check if it's a market event-based task
        if task.frequency in [ScheduleFrequency.MARKET_OPEN, ScheduleFrequency.MARKET_CLOSE]:
            return self._should_execute_market_event_task(task)
        
        return True
    
    def _should_execute_market_event_task(self, task: ScheduledTask) -> bool:
        """Check if market event-based task should execute"""
        current_status = self.schedule_state.current_market_status
        
        if task.frequency == ScheduleFrequency.MARKET_OPEN:
            return current_status == MarketStatus.OPEN
        elif task.frequency == ScheduleFrequency.MARKET_CLOSE:
            return current_status == MarketStatus.CLOSED
        
        return False
    
    async def execute_scheduled_task(self, task_name: str) -> Dict[str, Any]:
        """Execute a scheduled task with monitoring and error handling"""
        if not self.should_execute_task(task_name):
            return {
                'status': 'skipped',
                'reason': 'Task execution conditions not met',
                'task_name': task_name
            }
        
        task = self.scheduled_tasks[task_name]
        metrics = self.task_metrics[task_name]
        
        start_time = time_module.time()
        execution_result = {
            'status': 'failed',
            'task_name': task_name,
            'start_time': datetime.now(self.ist),
            'symbols_processed': 0,
            'errors': []
        }
        
        try:
            logger.info(f"Executing scheduled task: {task_name}")
            
            # Update metrics
            metrics.total_executions += 1
            metrics.last_execution = datetime.now(self.ist)
            
            # Execute the task function
            if task.function == "update_option_chains":
                result = await self._execute_option_chain_updates(task)
            elif task.function == "calculate_analytics":
                result = await self._execute_analytics_calculation(task)
            elif task.function == "market_open_initialization":
                result = await self._execute_market_open_init(task)
            elif task.function == "market_close_summary":
                result = await self._execute_market_close_summary(task)
            elif task.function == "monitor_performance":
                result = await self._execute_performance_monitoring(task)
            else:
                raise ValueError(f"Unknown task function: {task.function}")
            
            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000
            
            # Update success metrics
            if result.get('status') == 'success':
                execution_result['status'] = 'success'
                metrics.successful_executions += 1
                metrics.last_success = datetime.now(self.ist)
                metrics.consecutive_failures = 0
                
                # Update average duration
                if metrics.average_duration_ms == 0:
                    metrics.average_duration_ms = execution_time_ms
                else:
                    metrics.average_duration_ms = (
                        (metrics.average_duration_ms * (metrics.successful_executions - 1) + execution_time_ms)
                        / metrics.successful_executions
                    )
            else:
                metrics.failed_executions += 1
                metrics.last_failure = datetime.now(self.ist)
                metrics.consecutive_failures += 1
                execution_result['errors'] = result.get('errors', [])
            
            execution_result.update(result)
            execution_result['execution_time_ms'] = execution_time_ms
            
            # Check for alert conditions
            self._check_alert_conditions(task_name, metrics)
            
            logger.info(f"Task {task_name} completed: {execution_result['status']} in {execution_time_ms:.2f}ms")
            
        except Exception as e:
            # Handle execution errors
            execution_time_ms = (time_module.time() - start_time) * 1000
            error_message = str(e)
            
            metrics.failed_executions += 1
            metrics.last_failure = datetime.now(self.ist)
            metrics.consecutive_failures += 1
            metrics.errors.append(error_message)
            
            execution_result.update({
                'status': 'failed',
                'execution_time_ms': execution_time_ms,
                'errors': [error_message]
            })
            
            logger.error(f"Task {task_name} failed: {error_message}")
            
            # Check for alert conditions
            self._check_alert_conditions(task_name, metrics)
        
        return execution_result
    
    async def _execute_option_chain_updates(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute option chain data updates"""
        try:
            results = []
            symbols_processed = 0
            
            for symbol in task.symbols:
                result = await self.pipeline.process_symbol(symbol, is_equity=False)
                results.append(result)
                if result.status.value == 'success':
                    symbols_processed += 1
            
            return {
                'status': 'success',
                'symbols_processed': symbols_processed,
                'total_symbols': len(task.symbols),
                'results': [r.status.value for r in results]
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [str(e)],
                'symbols_processed': 0
            }
    
    async def _execute_analytics_calculation(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute analytics calculations"""
        try:
            calculations_completed = 0
            
            for symbol in task.symbols:
                # This would integrate with the analytics engine
                # For now, we'll simulate the calculation
                await asyncio.sleep(0.1)  # Simulate processing time
                calculations_completed += 1
            
            return {
                'status': 'success',
                'calculations_completed': calculations_completed,
                'symbols_processed': len(task.symbols)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [str(e)],
                'calculations_completed': 0
            }
    
    async def _execute_market_open_init(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute market opening initialization"""
        try:
            logger.info("Executing market open initialization")
            
            # Initialize daily metrics
            self._reset_daily_metrics()
            
            # Warm up data pipeline
            await self.pipeline.process_batch(task.symbols[:2])  # Test with first 2 symbols
            
            return {
                'status': 'success',
                'message': 'Market open initialization completed',
                'symbols_processed': len(task.symbols)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [str(e)]
            }
    
    async def _execute_market_close_summary(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute market close summary and cleanup"""
        try:
            logger.info("Executing market close summary")
            
            # Generate daily summary
            daily_summary = self._generate_daily_summary()
            
            # Save performance history
            self.performance_history.append({
                'date': datetime.now(self.ist).date().isoformat(),
                'summary': daily_summary,
                'timestamp': datetime.now(self.ist).isoformat()
            })
            
            return {
                'status': 'success',
                'message': 'Market close summary completed',
                'daily_summary': daily_summary
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [str(e)]
            }
    
    async def _execute_performance_monitoring(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute performance monitoring"""
        try:
            performance_report = self.get_performance_report()
            
            # Check for performance issues
            issues = self._detect_performance_issues(performance_report)
            
            if issues:
                logger.warning(f"Performance issues detected: {issues}")
            
            return {
                'status': 'success',
                'performance_report': performance_report,
                'issues_detected': len(issues),
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [str(e)]
            }
    
    def _check_alert_conditions(self, task_name: str, metrics: ScheduleMetrics):
        """Check if any alert conditions are met"""
        alerts = []
        
        # Check consecutive failures
        if metrics.consecutive_failures >= self.alert_thresholds['max_consecutive_failures']:
            alerts.append({
                'type': 'consecutive_failures',
                'task': task_name,
                'count': metrics.consecutive_failures,
                'severity': 'high'
            })
        
        # Check failure rate
        if metrics.total_executions > 0:
            failure_rate = metrics.failed_executions / metrics.total_executions
            if failure_rate > self.alert_thresholds['max_failure_rate']:
                alerts.append({
                    'type': 'high_failure_rate',
                    'task': task_name,
                    'rate': failure_rate,
                    'severity': 'medium'
                })
        
        # Check average duration
        if metrics.average_duration_ms > self.alert_thresholds['max_average_duration_ms']:
            alerts.append({
                'type': 'slow_execution',
                'task': task_name,
                'duration_ms': metrics.average_duration_ms,
                'severity': 'low'
            })
        
        # Send alerts if any are triggered
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification"""
        logger.warning(f"ALERT: {alert['type']} for task {alert['task']} - Severity: {alert['severity']}")
        # Here you would integrate with your alerting system
        # For now, we'll just log the alert
    
    def _reset_daily_metrics(self):
        """Reset daily metrics at market open"""
        for metrics in self.task_metrics.values():
            metrics.consecutive_failures = 0
            # Keep historical data but reset daily counters if needed
    
    def _generate_daily_summary(self) -> Dict[str, Any]:
        """Generate daily performance summary"""
        total_executions = sum(m.total_executions for m in self.task_metrics.values())
        total_successful = sum(m.successful_executions for m in self.task_metrics.values())
        
        return {
            'total_executions': total_executions,
            'successful_executions': total_successful,
            'success_rate': total_successful / total_executions if total_executions > 0 else 0,
            'tasks_summary': {
                name: {
                    'executions': metrics.total_executions,
                    'success_rate': metrics.successful_executions / metrics.total_executions
                    if metrics.total_executions > 0 else 0,
                    'avg_duration_ms': metrics.average_duration_ms
                }
                for name, metrics in self.task_metrics.items()
            }
        }
    
    def _detect_performance_issues(self, report: Dict[str, Any]) -> List[str]:
        """Detect performance issues from report"""
        issues = []
        
        for task_name, task_metrics in report.get('task_metrics', {}).items():
            if task_metrics['success_rate'] < self.alert_thresholds['min_success_rate']:
                issues.append(f"Low success rate for {task_name}: {task_metrics['success_rate']:.2%}")
            
            if task_metrics['avg_duration_ms'] > self.alert_thresholds['max_average_duration_ms']:
                issues.append(f"Slow execution for {task_name}: {task_metrics['avg_duration_ms']:.2f}ms")
        
        return issues
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'timestamp': datetime.now(self.ist).isoformat(),
            'schedule_state': {
                'is_active': self.schedule_state.is_active,
                'current_market_status': self.schedule_state.current_market_status.value
                if self.schedule_state.current_market_status else None,
                'active_tasks': self.schedule_state.active_tasks,
                'paused_tasks': self.schedule_state.paused_tasks
            },
            'task_metrics': {
                name: {
                    'total_executions': metrics.total_executions,
                    'successful_executions': metrics.successful_executions,
                    'failed_executions': metrics.failed_executions,
                    'success_rate': metrics.successful_executions / metrics.total_executions
                    if metrics.total_executions > 0 else 0,
                    'avg_duration_ms': metrics.average_duration_ms,
                    'consecutive_failures': metrics.consecutive_failures,
                    'last_execution': metrics.last_execution.isoformat()
                    if metrics.last_execution else None,
                    'last_success': metrics.last_success.isoformat()
                    if metrics.last_success else None
                }
                for name, metrics in self.task_metrics.items()
            },
            'scheduled_tasks': {
                name: {
                    'frequency': task.frequency.value,
                    'priority': task.priority.value,
                    'enabled': task.enabled,
                    'symbols_count': len(task.symbols)
                }
                for name, task in self.scheduled_tasks.items()
            }
        }
    
    def start_scheduler(self):
        """Start the scheduling system"""
        self.schedule_state.is_active = True
        self.update_market_status()
        logger.info("Market Hours Scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduling system"""
        self.schedule_state.is_active = False
        logger.info("Market Hours Scheduler stopped")
    
    def get_next_execution_times(self) -> Dict[str, datetime]:
        """Get next execution times for all scheduled tasks"""
        # This would calculate based on frequency and current time
        # For now, return a placeholder
        return {
            name: datetime.now(self.ist) + timedelta(minutes=5)
            for name in self.scheduled_tasks.keys()
        }

# Global scheduler instance
market_scheduler = MarketHoursScheduler()

# Celery task definitions for scheduled execution
@celery_app.task(bind=True)
def execute_scheduled_task(self, task_name: str):
    """Celery task wrapper for scheduled task execution"""
    try:
        # Run the async task in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(market_scheduler.execute_scheduled_task(task_name))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Celery task execution failed for {task_name}: {str(e)}")
        raise

@celery_app.task
def update_market_status():
    """Celery task to update market status"""
    market_scheduler.update_market_status()
    return {'status': 'success', 'timestamp': datetime.now().isoformat()}

@celery_app.task
def generate_performance_report():
    """Celery task to generate performance report"""
    return market_scheduler.get_performance_report()

# Schedule configuration for Celery Beat
def get_celery_schedule() -> Dict[str, Any]:
    """Get Celery Beat schedule configuration"""
    return {
        # Update market status every minute
        'update-market-status': {
            'task': 'app.services.market_scheduler.update_market_status',
            'schedule': crontab(minute='*'),
        },
        
        # Option chain updates every 5 minutes during market hours
        'option-chain-updates': {
            'task': 'app.services.market_scheduler.execute_scheduled_task',
            'schedule': crontab(minute='*/5'),
            'args': ('option_chain_updates',),
        },
        
        # Analytics calculation every 15 minutes during market hours
        'analytics-calculation': {
            'task': 'app.services.market_scheduler.execute_scheduled_task',
            'schedule': crontab(minute='*/15'),
            'args': ('analytics_calculation',),
        },
        
        # Performance monitoring every hour
        'performance-monitoring': {
            'task': 'app.services.market_scheduler.execute_scheduled_task',
            'schedule': crontab(minute=0),
            'args': ('performance_monitoring',),
        },
        
        # Market open initialization at 9:15 AM IST
        'market-open-init': {
            'task': 'app.services.market_scheduler.execute_scheduled_task',
            'schedule': crontab(hour=9, minute=15),
            'args': ('market_open_init',),
        },
        
        # Market close summary at 3:30 PM IST
        'market-close-summary': {
            'task': 'app.services.market_scheduler.execute_scheduled_task',
            'schedule': crontab(hour=15, minute=30),
            'args': ('market_close_summary',),
        },
    }

# Convenience functions
def start_market_scheduler():
    """Start the market scheduler"""
    market_scheduler.start_scheduler()

def stop_market_scheduler():
    """Stop the market scheduler"""
    market_scheduler.stop_scheduler()

def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status"""
    return market_scheduler.get_performance_report()

def add_custom_schedule(task: ScheduledTask):
    """Add a custom scheduled task"""
    market_scheduler.add_scheduled_task(task) 