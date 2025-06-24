"""
Password Manager for secure password hashing and verification.

Uses bcrypt for password hashing with salt and provides utilities
for password strength validation and secure password handling.
"""

import bcrypt
import re
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext


class PasswordStrengthError(Exception):
    """Exception raised when password doesn't meet strength requirements."""
    pass


class PasswordManager:
    """Password management class with bcrypt hashing and validation."""
    
    def __init__(self, rounds: int = 12):
        """
        Initialize password manager with bcrypt configuration.
        
        Args:
            rounds: Bcrypt hashing rounds (higher = more secure but slower)
        """
        self.rounds = rounds
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password string
        """
        # Convert to bytes if string
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password, salt)
        
        # Return as string
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            hashed_password: Hashed password to check against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Convert to bytes if needed
            if isinstance(password, str):
                password = password.encode('utf-8')
            if isinstance(hashed_password, str):
                hashed_password = hashed_password.encode('utf-8')
            
            return bcrypt.checkpw(password, hashed_password)
        except (ValueError, TypeError):
            return False
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength against security requirements.
        
        Args:
            password: Password to validate
            
        Returns:
            Dictionary with validation results and suggestions
        """
        errors = []
        suggestions = []
        score = 0
        
        # Minimum length check
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        else:
            score += 1
        
        # Maximum length check (prevent DoS)
        if len(password) > 128:
            errors.append("Password must be less than 128 characters")
        
        # Character type checks
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*()_+=\-\[\]{};:"\|,.<>?]', password))
        
        if not has_lower:
            errors.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if not has_upper:
            errors.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if not has_digit:
            errors.append("Password must contain at least one number")
        else:
            score += 1
        
        if not has_special:
            suggestions.append("Consider adding special characters for stronger security")
        else:
            score += 1
        
        # Common password checks
        common_passwords = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey'
        ]
        
        if password.lower() in common_passwords:
            errors.append("Password is too common. Please choose a unique password")
        
        # Sequential character check
        if self._has_sequential_chars(password):
            suggestions.append("Avoid sequential characters like 'abc' or '123'")
        
        # Repeated character check
        if self._has_repeated_chars(password):
            suggestions.append("Avoid repeated characters like 'aaa' or '111'")
        
        # Calculate strength
        if score >= 5:
            strength = "strong"
        elif score >= 3:
            strength = "medium"
        else:
            strength = "weak"
        
        return {
            "is_valid": len(errors) == 0,
            "strength": strength,
            "score": score,
            "errors": errors,
            "suggestions": suggestions
        }
    
    def _has_sequential_chars(self, password: str) -> bool:
        """Check for sequential characters in password."""
        sequences = ['abc', 'bcd', 'cde', 'def', '123', '234', '345', '456']
        password_lower = password.lower()
        
        for seq in sequences:
            if seq in password_lower:
                return True
        
        return False
    
    def _has_repeated_chars(self, password: str) -> bool:
        """Check for repeated characters in password."""
        for i in range(len(password) - 2):
            if password[i] == password[i+1] == password[i+2]:
                return True
        return False
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Length of password to generate
            
        Returns:
            Secure random password string
        """
        import secrets
        import string
        
        # Character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-="
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill remaining length with random characters
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)


# Global password manager instance
password_manager = PasswordManager()


def hash_password(password: str) -> str:
    """
    Hash a password using the global password manager.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    return password_manager.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password using the global password manager.
    
    Args:
        password: Plain text password
        hashed_password: Hashed password to verify against
        
    Returns:
        True if password matches, False otherwise
    """
    return password_manager.verify_password(password, hashed_password)


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength using the global password manager.
    
    Args:
        password: Password to validate
        
    Returns:
        Validation results dictionary
    """
    return password_manager.validate_password_strength(password)


def generate_secure_password(length: int = 16) -> str:
    """
    Generate a secure password using the global password manager.
    
    Args:
        length: Length of password to generate
        
    Returns:
        Secure random password
    """
    return password_manager.generate_secure_password(length)


def require_strong_password(password: str) -> None:
    """
    Validate password strength and raise exception if not strong enough.
    
    Args:
        password: Password to validate
        
    Raises:
        PasswordStrengthError: If password doesn't meet requirements
    """
    validation = validate_password_strength(password)
    
    if not validation["is_valid"]:
        error_message = "Password does not meet security requirements:\n" + \
                       "\n".join(f"- {error}" for error in validation["errors"])
        raise PasswordStrengthError(error_message) 