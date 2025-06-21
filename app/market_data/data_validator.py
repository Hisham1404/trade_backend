"""
Data Validation Module

This module provides data quality checks and validation for option chain data.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    warnings: List[str]
    errors: List[str]
    anomalies: List[str]
    metrics: Dict[str, Any]

class OptionDataValidator:
    """Validates option chain data for quality and anomalies"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_option_count': 10,
            'max_bid_ask_spread_pct': 50.0,
            'min_strikes_count': 5,
            'max_price_change_pct': 200.0
        }
    
    def validate_option_chain_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate option chain data"""
        warnings = []
        errors = []
        anomalies = []
        metrics = {}
        
        try:
            # Basic structure validation
            if not isinstance(data, dict):
                errors.append("Data is not a dictionary")
                return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
            
            if 'records' not in data:
                errors.append("Missing 'records' key in data")
                return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
            
            records = data['records']
            if 'data' not in records:
                errors.append("Missing 'data' key in records")
                return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
            
            option_data = records['data']
            if not isinstance(option_data, list):
                errors.append("Option data is not a list")
                return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
            
            if len(option_data) == 0:
                errors.append("Option data is empty")
                return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
            
            # Calculate metrics
            metrics = self._calculate_metrics(option_data, warnings, errors, anomalies)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(warnings, errors, anomalies)
            
            is_valid = len(errors) == 0
            
            return self._create_result(is_valid, quality_score, warnings, errors, anomalies, metrics)
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return self._create_result(False, 0.0, warnings, errors, anomalies, metrics)
    
    def _calculate_metrics(self, option_data: List[Dict], warnings: List[str], 
                          errors: List[str], anomalies: List[str]) -> Dict[str, Any]:
        """Calculate key metrics from option data"""
        metrics = {
            'total_options': len(option_data),
            'call_count': 0,
            'put_count': 0,
            'total_call_oi': 0,
            'total_put_oi': 0,
            'put_call_ratio': 0.0,
            'avg_iv': 0.0
        }
        
        iv_values = []
        
        for item in option_data:
            if 'CE' in item:
                metrics['call_count'] += 1
                ce_oi = item['CE'].get('openInterest', 0)
                metrics['total_call_oi'] += ce_oi
                
                iv = item['CE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_values.append(iv)
            
            if 'PE' in item:
                metrics['put_count'] += 1
                pe_oi = item['PE'].get('openInterest', 0)
                metrics['total_put_oi'] += pe_oi
                
                iv = item['PE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_values.append(iv)
        
        # Calculate PCR
        if metrics['total_call_oi'] > 0:
            metrics['put_call_ratio'] = metrics['total_put_oi'] / metrics['total_call_oi']
        
        # Calculate average IV
        if iv_values:
            metrics['avg_iv'] = np.mean(iv_values)
        
        # Validation checks
        if metrics['call_count'] == 0:
            errors.append("No call options found")
        
        if metrics['put_count'] == 0:
            errors.append("No put options found")
        
        if metrics['total_options'] < self.quality_thresholds['min_option_count']:
            warnings.append(f"Low option count: {metrics['total_options']}")
        
        return metrics
    
    def _calculate_quality_score(self, warnings: List[str], errors: List[str], 
                               anomalies: List[str]) -> float:
        """Calculate quality score based on issues found"""
        score = 1.0
        
        # Deduct for errors
        score -= len(errors) * 0.2
        
        # Deduct for warnings
        score -= len(warnings) * 0.05
        
        # Deduct for anomalies
        score -= len(anomalies) * 0.02
        
        return max(0.0, min(1.0, score))
    
    def _create_result(self, is_valid: bool, quality_score: float, 
                      warnings: List[str], errors: List[str], 
                      anomalies: List[str], metrics: Dict[str, Any]) -> ValidationResult:
        """Create validation result object"""
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            warnings=warnings,
            errors=errors,
            anomalies=anomalies,
            metrics=metrics
        )

# Convenience function
def validate_option_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate option chain data"""
    validator = OptionDataValidator()
    return validator.validate_option_chain_data(data)

