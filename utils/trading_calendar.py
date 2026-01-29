# utils/trading_calendar.py
"""
Trading Calendar Utility

Provides functions for validating U.S. trading days and finding
the next/previous trading day relative to a given date.

Uses pandas USFederalHolidayCalendar for accurate holiday detection.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# Singleton calendar instance
US_HOLIDAY_CAL = USFederalHolidayCalendar()


def is_us_business_day(ts) -> bool:
    """
    Return True when the timestamp falls on a U.S. business day.
    
    Excludes:
    - Weekends (Saturday, Sunday)
    - US Federal holidays
    
    Args:
        ts: datetime, date, or pandas Timestamp
        
    Returns:
        True if it's a valid trading day
    """
    normalized = pd.Timestamp(ts).normalize()
    
    # Weekend check (5 = Saturday, 6 = Sunday)
    if normalized.weekday() >= 5:
        return False
    
    # Holiday check
    holidays = US_HOLIDAY_CAL.holidays(start=normalized, end=normalized)
    return normalized not in holidays


def next_trading_day(start_dt) -> datetime:
    """
    Advance to the next available U.S. trading day on or after the supplied datetime.
    
    If start_dt is already a trading day, returns start_dt.
    
    Args:
        start_dt: Starting datetime
        
    Returns:
        Next trading day as datetime
        
    Raises:
        ValueError: If no trading day found within 1 year
    """
    dt = pd.Timestamp(start_dt).normalize()
    guard = 0
    
    while not is_us_business_day(dt):
        dt += timedelta(days=1)
        guard += 1
        if guard > 366:
            raise ValueError("Unable to resolve the next trading day within one year.")
    
    return dt.to_pydatetime()


def previous_trading_day(start_dt) -> datetime:
    """
    Roll backward to the prior U.S. trading day before the supplied datetime.
    
    Args:
        start_dt: Starting datetime
        
    Returns:
        Previous trading day as datetime
        
    Raises:
        ValueError: If no trading day found within 1 year
    """
    dt = pd.Timestamp(start_dt).normalize() - timedelta(days=1)
    guard = 0
    
    while not is_us_business_day(dt):
        dt -= timedelta(days=1)
        guard += 1
        if guard > 366:
            raise ValueError("Unable to resolve the previous trading day within one year.")
    
    return dt.to_pydatetime()


def get_trading_dates(selected_date: datetime) -> tuple[datetime, datetime]:
    """
    Get execution and analysis dates for a selected date.
    
    Args:
        selected_date: User-selected date
        
    Returns:
        Tuple of (execution_date, analysis_date)
        - execution_date: Next trading day for trade execution
        - analysis_date: Previous trading day for data analysis
    """
    execution_date = next_trading_day(selected_date)
    
    try:
        analysis_date = previous_trading_day(execution_date)
    except ValueError:
        analysis_date = execution_date
    
    return execution_date, analysis_date
