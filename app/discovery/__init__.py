from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source_finder import SourceDiscovery
    from .source_categorizer import SourceCategorizer
    from .validation_scoring import ValidationScoringSystem
else:
    # Discovery module initialization
    try:
        from .source_finder import SourceDiscovery
        from .source_categorizer import SourceCategorizer
        from .validation_scoring import ValidationScoringSystem
    except ImportError:
        SourceDiscovery = object
        SourceCategorizer = object
        ValidationScoringSystem = object

from .source_manager import SourceManager
from .source_lifecycle import (
    SourceLifecycleManager,
    SourceHealthMonitor,
    ContentFreshnessMonitor,
    SourceMaintainer,
    SourceArchiver,
    LifecycleNotifier,
    HealthStatus,
    SourceHealthCheck,
    ContentFreshnessReport,
    LifecycleEvent,
    run_lifecycle_maintenance,
    get_source_health_report
)

__all__ = [
    "SourceDiscovery", 
    "SourceManager", 
    "SourceCategorizer",
    "ValidationScoringSystem",
    "SourceLifecycleManager",
    "SourceHealthMonitor",
    "ContentFreshnessMonitor",
    "SourceMaintainer",
    "SourceArchiver",
    "LifecycleNotifier",
    "HealthStatus",
    "SourceHealthCheck",
    "ContentFreshnessReport",
    "LifecycleEvent",
    "run_lifecycle_maintenance",
    "get_source_health_report"
] 