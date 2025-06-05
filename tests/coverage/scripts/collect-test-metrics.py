#!/usr/bin/env python3
"""
Comprehensive Test Metrics Collection Script

This script aggregates coverage data, performance benchmarks, test execution statistics,
and quality indicators into unified reporting for flyrigloader testing infrastructure.
Provides detailed analytics for development team insights and automated quality assurance
monitoring per Section 3.6.4 quality metrics dashboard integration.

Requirements Coverage:
- Section 3.6.4: Quality metrics dashboard integration with coverage trend tracking
- Section 0.2.5: Test execution time monitoring and failure rate analysis
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- Section 2.1.12: Coverage Enhancement System with detailed reporting and visualization
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
from loguru import logger


class TestMetricsCollector:
    """
    Comprehensive test metrics collection and aggregation system.
    
    Aggregates coverage data, performance benchmarks, test execution statistics,
    and quality indicators into unified reporting for development team insights
    and automated quality assurance monitoring.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        metrics_output_dir: Optional[Path] = None,
        include_historical: bool = True
    ):
        """
        Initialize test metrics collector.
        
        Args:
            base_dir: Base directory for test results (defaults to project root)
            metrics_output_dir: Output directory for aggregated metrics
            include_historical: Whether to include historical trend analysis
        """
        self.base_dir = base_dir or Path.cwd()
        self.metrics_output_dir = metrics_output_dir or (self.base_dir / "tests" / "coverage" / "metrics")
        self.include_historical = include_historical
        
        # Ensure output directory exists
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.coverage_metrics: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.test_execution_metrics: Dict[str, Any] = {}
        self.quality_indicators: Dict[str, Any] = {}
        
        # Load configuration files
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load configuration files for thresholds and quality gates."""
        try:
            # Load coverage thresholds
            thresholds_path = self.base_dir / "tests" / "coverage" / "coverage-thresholds.json"
            if thresholds_path.exists():
                with open(thresholds_path, 'r') as f:
                    self.coverage_thresholds = json.load(f)
            else:
                logger.warning(f"Coverage thresholds file not found: {thresholds_path}")
                self.coverage_thresholds = {}

            # Load quality gates configuration
            quality_gates_path = self.base_dir / "tests" / "coverage" / "quality-gates.yml"
            if quality_gates_path.exists():
                import yaml
                with open(quality_gates_path, 'r') as f:
                    self.quality_gates = yaml.safe_load(f)
            else:
                logger.warning(f"Quality gates file not found: {quality_gates_path}")
                self.quality_gates = {}

        except Exception as e:
            logger.warning(f"Error loading configurations: {e}")
            self.coverage_thresholds = {}
            self.quality_gates = {}

    def collect_coverage_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive coverage metrics from various sources.
        
        Returns:
            Dict containing coverage statistics, module breakdowns, and trends
        """
        logger.info("Collecting coverage metrics...")
        
        coverage_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_coverage": {},
            "module_coverage": {},
            "branch_coverage": {},
            "trend_analysis": {},
            "quality_status": {}
        }

        try:
            # Collect from XML coverage report
            xml_data = self._parse_coverage_xml()
            if xml_data:
                coverage_data["overall_coverage"].update(xml_data["overall"])
                coverage_data["module_coverage"].update(xml_data["modules"])

            # Collect from JSON coverage report
            json_data = self._parse_coverage_json()
            if json_data:
                coverage_data["branch_coverage"] = json_data.get("branch_coverage", {})
                coverage_data["overall_coverage"]["lines_missed"] = json_data.get("totals", {}).get("missing_lines", 0)

            # Calculate quality status against thresholds
            coverage_data["quality_status"] = self._evaluate_coverage_quality(coverage_data)

            # Collect historical trends if enabled
            if self.include_historical:
                coverage_data["trend_analysis"] = self._analyze_coverage_trends(coverage_data)

        except Exception as e:
            logger.error(f"Error collecting coverage metrics: {e}")
            coverage_data["error"] = str(e)

        self.coverage_metrics = coverage_data
        return coverage_data

    def _parse_coverage_xml(self) -> Optional[Dict[str, Any]]:
        """Parse XML coverage report for metrics extraction."""
        xml_path = self.base_dir / "coverage.xml"
        if not xml_path.exists():
            logger.warning(f"Coverage XML not found: {xml_path}")
            return None

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract overall coverage
            overall = {}
            if root.attrib:
                overall["line_rate"] = float(root.attrib.get("line-rate", 0)) * 100
                overall["branch_rate"] = float(root.attrib.get("branch-rate", 0)) * 100
                overall["lines_covered"] = int(root.attrib.get("lines-covered", 0))
                overall["lines_valid"] = int(root.attrib.get("lines-valid", 0))
                overall["branches_covered"] = int(root.attrib.get("branches-covered", 0))
                overall["branches_valid"] = int(root.attrib.get("branches-valid", 0))

            # Extract module-specific coverage
            modules = {}
            for package in root.findall(".//package"):
                package_name = package.attrib.get("name", "unknown")
                
                for cls in package.findall("classes/class"):
                    filename = cls.attrib.get("filename", "")
                    if "src/flyrigloader" in filename:
                        module_name = filename.replace("src/", "").replace("/", ".")
                        if module_name.endswith(".py"):
                            module_name = module_name[:-3]
                        
                        modules[module_name] = {
                            "line_rate": float(cls.attrib.get("line-rate", 0)) * 100,
                            "branch_rate": float(cls.attrib.get("branch-rate", 0)) * 100,
                            "lines_covered": len([l for l in cls.findall("lines/line") if l.attrib.get("hits", "0") != "0"]),
                            "lines_total": len(cls.findall("lines/line"))
                        }

            return {"overall": overall, "modules": modules}

        except Exception as e:
            logger.error(f"Error parsing coverage XML: {e}")
            return None

    def _parse_coverage_json(self) -> Optional[Dict[str, Any]]:
        """Parse JSON coverage report for detailed metrics."""
        json_path = self.base_dir / "coverage.json"
        if not json_path.exists():
            logger.warning(f"Coverage JSON not found: {json_path}")
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract branch coverage details
            branch_coverage = {}
            files = data.get("files", {})
            
            for filename, file_data in files.items():
                if "src/flyrigloader" in filename:
                    module_name = filename.replace("src/", "").replace("/", ".")
                    if module_name.endswith(".py"):
                        module_name = module_name[:-3]
                    
                    summary = file_data.get("summary", {})
                    branch_coverage[module_name] = {
                        "covered_branches": summary.get("covered_branches", 0),
                        "num_branches": summary.get("num_branches", 0),
                        "missing_branches": summary.get("missing_branches", []),
                        "branch_rate": (summary.get("covered_branches", 0) / max(1, summary.get("num_branches", 1))) * 100
                    }

            return {
                "branch_coverage": branch_coverage,
                "totals": data.get("totals", {})
            }

        except Exception as e:
            logger.error(f"Error parsing coverage JSON: {e}")
            return None

    def _evaluate_coverage_quality(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate coverage quality against configured thresholds."""
        quality_status = {
            "overall_status": "unknown",
            "critical_modules_status": "unknown",
            "violations": [],
            "warnings": []
        }

        try:
            # Check overall coverage threshold
            overall_threshold = self.coverage_thresholds.get("global_configuration", {}).get("overall_threshold", 90.0)
            overall_coverage = coverage_data.get("overall_coverage", {}).get("line_rate", 0)
            
            if overall_coverage >= overall_threshold:
                quality_status["overall_status"] = "pass"
            else:
                quality_status["overall_status"] = "fail"
                quality_status["violations"].append(
                    f"Overall coverage {overall_coverage:.1f}% below threshold {overall_threshold}%"
                )

            # Check critical module coverage
            critical_modules = self.coverage_thresholds.get("module_thresholds", {}).get("critical_modules", {}).get("modules", {})
            module_coverage = coverage_data.get("module_coverage", {})
            
            critical_violations = []
            for module_path, module_config in critical_modules.items():
                module_name = module_path.replace("src/", "").replace("/", ".").replace(".py", "")
                threshold = module_config.get("threshold", 100.0)
                
                actual_coverage = module_coverage.get(module_name, {}).get("line_rate", 0)
                if actual_coverage < threshold:
                    critical_violations.append(
                        f"Critical module {module_name}: {actual_coverage:.1f}% < {threshold}%"
                    )

            if critical_violations:
                quality_status["critical_modules_status"] = "fail"
                quality_status["violations"].extend(critical_violations)
            else:
                quality_status["critical_modules_status"] = "pass"

        except Exception as e:
            logger.error(f"Error evaluating coverage quality: {e}")
            quality_status["error"] = str(e)

        return quality_status

    def _analyze_coverage_trends(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage trends over time for regression detection."""
        trend_analysis = {
            "trend_direction": "stable",
            "regression_detected": False,
            "historical_comparison": {},
            "confidence_metrics": {}
        }

        try:
            # Load historical metrics if available
            historical_file = self.metrics_output_dir / "historical_coverage.json"
            historical_data = []
            
            if historical_file.exists():
                with open(historical_file, 'r') as f:
                    historical_data = json.load(f)

            # Add current data to historical record
            historical_data.append({
                "timestamp": current_data["timestamp"],
                "overall_coverage": current_data.get("overall_coverage", {}).get("line_rate", 0),
                "critical_modules_avg": self._calculate_critical_modules_average(current_data)
            })

            # Keep only last 30 data points for trend analysis
            historical_data = historical_data[-30:]

            # Perform trend analysis
            if len(historical_data) >= 3:
                recent_avg = sum(d["overall_coverage"] for d in historical_data[-3:]) / 3
                older_avg = sum(d["overall_coverage"] for d in historical_data[-6:-3]) / 3 if len(historical_data) >= 6 else recent_avg
                
                trend_analysis["trend_direction"] = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                trend_analysis["regression_detected"] = recent_avg < (older_avg - 2.0)  # 2% regression threshold
                
                trend_analysis["historical_comparison"] = {
                    "recent_average": recent_avg,
                    "older_average": older_avg,
                    "data_points": len(historical_data)
                }

            # Save updated historical data
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing coverage trends: {e}")
            trend_analysis["error"] = str(e)

        return trend_analysis

    def _calculate_critical_modules_average(self, coverage_data: Dict[str, Any]) -> float:
        """Calculate average coverage for critical modules."""
        critical_modules = self.coverage_thresholds.get("module_thresholds", {}).get("critical_modules", {}).get("modules", {})
        module_coverage = coverage_data.get("module_coverage", {})
        
        coverages = []
        for module_path in critical_modules.keys():
            module_name = module_path.replace("src/", "").replace("/", ".").replace(".py", "")
            coverage = module_coverage.get(module_name, {}).get("line_rate", 0)
            if coverage > 0:
                coverages.append(coverage)
        
        return sum(coverages) / len(coverages) if coverages else 0.0

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect performance benchmark metrics from pytest-benchmark results.
        
        Returns:
            Dict containing performance statistics and SLA compliance data
        """
        logger.info("Collecting performance metrics...")
        
        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_results": {},
            "sla_compliance": {},
            "performance_trends": {},
            "regression_analysis": {}
        }

        try:
            # Parse pytest-benchmark results
            benchmark_data = self._parse_benchmark_results()
            if benchmark_data:
                performance_data["benchmark_results"] = benchmark_data

            # Evaluate SLA compliance
            performance_data["sla_compliance"] = self._evaluate_performance_slas(benchmark_data)

            # Analyze performance trends
            if self.include_historical:
                performance_data["performance_trends"] = self._analyze_performance_trends(benchmark_data)

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            performance_data["error"] = str(e)

        self.performance_metrics = performance_data
        return performance_data

    def _parse_benchmark_results(self) -> Dict[str, Any]:
        """Parse pytest-benchmark results for performance metrics."""
        benchmark_results = {}
        
        # Look for benchmark JSON files
        benchmark_files = list(self.base_dir.glob(".benchmarks/**/*.json"))
        
        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                
                benchmarks = data.get("benchmarks", [])
                for benchmark in benchmarks:
                    name = benchmark.get("name", "unknown")
                    stats = benchmark.get("stats", {})
                    
                    benchmark_results[name] = {
                        "mean": stats.get("mean", 0),
                        "median": stats.get("median", 0),
                        "min": stats.get("min", 0),
                        "max": stats.get("max", 0),
                        "stddev": stats.get("stddev", 0),
                        "rounds": stats.get("rounds", 0),
                        "ops": stats.get("ops", 0) if stats.get("ops") else (1 / stats.get("mean", 1))
                    }

            except Exception as e:
                logger.warning(f"Error parsing benchmark file {benchmark_file}: {e}")

        return benchmark_results

    def _evaluate_performance_slas(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance against defined SLAs."""
        sla_compliance = {
            "data_loading_sla": {"status": "unknown", "violations": []},
            "dataframe_transformation_sla": {"status": "unknown", "violations": []},
            "overall_status": "unknown"
        }

        try:
            # Check data loading SLA: 1 second per 100MB
            data_loading_benchmarks = {name: data for name, data in benchmark_data.items() 
                                     if "load" in name.lower() or "pickle" in name.lower()}
            
            data_loading_violations = []
            for name, data in data_loading_benchmarks.items():
                # Assume benchmarks test 100MB equivalent
                if data["mean"] > 1.0:  # 1 second threshold
                    data_loading_violations.append(f"{name}: {data['mean']:.3f}s > 1.0s")
            
            sla_compliance["data_loading_sla"]["status"] = "pass" if not data_loading_violations else "fail"
            sla_compliance["data_loading_sla"]["violations"] = data_loading_violations

            # Check DataFrame transformation SLA: 500ms per 1M rows
            transform_benchmarks = {name: data for name, data in benchmark_data.items() 
                                  if "transform" in name.lower() or "dataframe" in name.lower()}
            
            transform_violations = []
            for name, data in transform_benchmarks.items():
                # Assume benchmarks test 1M row equivalent
                if data["mean"] > 0.5:  # 500ms threshold
                    transform_violations.append(f"{name}: {data['mean']:.3f}s > 0.5s")
            
            sla_compliance["dataframe_transformation_sla"]["status"] = "pass" if not transform_violations else "fail"
            sla_compliance["dataframe_transformation_sla"]["violations"] = transform_violations

            # Overall SLA status
            all_violations = data_loading_violations + transform_violations
            sla_compliance["overall_status"] = "pass" if not all_violations else "fail"

        except Exception as e:
            logger.error(f"Error evaluating performance SLAs: {e}")
            sla_compliance["error"] = str(e)

        return sla_compliance

    def _analyze_performance_trends(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends for regression detection."""
        trend_analysis = {
            "regression_detected": False,
            "performance_changes": {},
            "trend_summary": {}
        }

        try:
            # Load historical performance data
            historical_file = self.metrics_output_dir / "historical_performance.json"
            historical_data = []
            
            if historical_file.exists():
                with open(historical_file, 'r') as f:
                    historical_data = json.load(f)

            # Add current data
            current_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "benchmarks": benchmark_data
            }
            historical_data.append(current_entry)
            
            # Keep last 20 entries
            historical_data = historical_data[-20:]

            # Analyze trends for each benchmark
            if len(historical_data) >= 3:
                for benchmark_name in benchmark_data.keys():
                    recent_values = []
                    for entry in historical_data[-3:]:
                        if benchmark_name in entry.get("benchmarks", {}):
                            recent_values.append(entry["benchmarks"][benchmark_name]["mean"])
                    
                    older_values = []
                    for entry in historical_data[-6:-3] if len(historical_data) >= 6 else []:
                        if benchmark_name in entry.get("benchmarks", {}):
                            older_values.append(entry["benchmarks"][benchmark_name]["mean"])
                    
                    if recent_values and older_values:
                        recent_avg = sum(recent_values) / len(recent_values)
                        older_avg = sum(older_values) / len(older_values)
                        
                        # Detect regression (>20% performance decrease)
                        if recent_avg > older_avg * 1.2:
                            trend_analysis["regression_detected"] = True
                            trend_analysis["performance_changes"][benchmark_name] = {
                                "status": "regression",
                                "recent_avg": recent_avg,
                                "older_avg": older_avg,
                                "change_percent": ((recent_avg - older_avg) / older_avg) * 100
                            }

            # Save updated historical data
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            trend_analysis["error"] = str(e)

        return trend_analysis

    def collect_test_execution_metrics(self) -> Dict[str, Any]:
        """
        Collect test execution statistics and categorization data.
        
        Returns:
            Dict containing test execution times, pass/fail rates, and categorization
        """
        logger.info("Collecting test execution metrics...")
        
        execution_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_summary": {},
            "test_categorization": {},
            "failure_analysis": {},
            "timing_analysis": {}
        }

        try:
            # Parse pytest execution results
            junit_data = self._parse_junit_results()
            if junit_data:
                execution_data["execution_summary"] = junit_data["summary"]
                execution_data["test_categorization"] = junit_data["categorization"]
                execution_data["failure_analysis"] = junit_data["failures"]
                execution_data["timing_analysis"] = junit_data["timing"]

        except Exception as e:
            logger.error(f"Error collecting test execution metrics: {e}")
            execution_data["error"] = str(e)

        self.test_execution_metrics = execution_data
        return execution_data

    def _parse_junit_results(self) -> Optional[Dict[str, Any]]:
        """Parse JUnit XML results for test execution metrics."""
        junit_path = self.base_dir / "junit.xml"
        if not junit_path.exists():
            logger.warning(f"JUnit XML not found: {junit_path}")
            return None

        try:
            tree = ET.parse(junit_path)
            root = tree.getroot()

            # Extract summary statistics
            summary = {
                "total_tests": int(root.attrib.get("tests", 0)),
                "failures": int(root.attrib.get("failures", 0)),
                "errors": int(root.attrib.get("errors", 0)),
                "skipped": int(root.attrib.get("skipped", 0)),
                "time": float(root.attrib.get("time", 0))
            }
            summary["passed"] = summary["total_tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
            summary["pass_rate"] = (summary["passed"] / max(1, summary["total_tests"])) * 100

            # Categorize tests by markers/names
            categorization = {
                "unit": 0,
                "integration": 0,
                "benchmark": 0,
                "other": 0
            }

            # Analyze failures
            failures = []
            timing_data = []

            for testcase in root.findall(".//testcase"):
                test_name = testcase.attrib.get("name", "")
                test_time = float(testcase.attrib.get("time", 0))
                classname = testcase.attrib.get("classname", "")

                # Categorize test
                if "test_unit" in test_name or "unit" in classname:
                    categorization["unit"] += 1
                elif "test_integration" in test_name or "integration" in classname:
                    categorization["integration"] += 1
                elif "test_benchmark" in test_name or "benchmark" in classname:
                    categorization["benchmark"] += 1
                else:
                    categorization["other"] += 1

                # Collect timing data
                timing_data.append({
                    "test_name": test_name,
                    "time": test_time,
                    "category": self._categorize_test(test_name, classname)
                })

                # Analyze failures
                failure = testcase.find("failure")
                error = testcase.find("error")
                if failure is not None or error is not None:
                    failure_info = failure if failure is not None else error
                    failures.append({
                        "test_name": test_name,
                        "type": failure_info.attrib.get("type", "unknown"),
                        "message": failure_info.attrib.get("message", ""),
                        "category": self._categorize_test(test_name, classname)
                    })

            # Analyze timing patterns
            timing_analysis = self._analyze_test_timing(timing_data)

            return {
                "summary": summary,
                "categorization": categorization,
                "failures": failures,
                "timing": timing_analysis
            }

        except Exception as e:
            logger.error(f"Error parsing JUnit XML: {e}")
            return None

    def _categorize_test(self, test_name: str, classname: str) -> str:
        """Categorize a test based on its name and class."""
        name_and_class = (test_name + " " + classname).lower()
        
        if "unit" in name_and_class or "test_unit" in name_and_class:
            return "unit"
        elif "integration" in name_and_class or "test_integration" in name_and_class:
            return "integration"
        elif "benchmark" in name_and_class or "test_benchmark" in name_and_class:
            return "benchmark"
        else:
            return "other"

    def _analyze_test_timing(self, timing_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test execution timing patterns."""
        if not timing_data:
            return {}

        df = pd.DataFrame(timing_data)
        
        timing_analysis = {
            "total_time": df["time"].sum(),
            "average_time": df["time"].mean(),
            "median_time": df["time"].median(),
            "slowest_tests": df.nlargest(5, "time")[["test_name", "time", "category"]].to_dict("records"),
            "category_timing": {}
        }

        # Analyze by category
        for category in df["category"].unique():
            category_data = df[df["category"] == category]
            timing_analysis["category_timing"][category] = {
                "count": len(category_data),
                "total_time": category_data["time"].sum(),
                "average_time": category_data["time"].mean(),
                "median_time": category_data["time"].median()
            }

        return timing_analysis

    def calculate_quality_indicators(self) -> Dict[str, Any]:
        """
        Calculate comprehensive quality indicators from all collected metrics.
        
        Returns:
            Dict containing quality scores, trends, and actionable insights
        """
        logger.info("Calculating quality indicators...")
        
        quality_indicators = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_quality_score": 0.0,
            "component_scores": {},
            "trend_indicators": {},
            "actionable_insights": [],
            "quality_status": "unknown"
        }

        try:
            # Calculate component scores
            coverage_score = self._calculate_coverage_score()
            performance_score = self._calculate_performance_score()
            execution_score = self._calculate_execution_score()

            quality_indicators["component_scores"] = {
                "coverage": coverage_score,
                "performance": performance_score,
                "execution": execution_score
            }

            # Calculate overall quality score (weighted average)
            weights = {"coverage": 0.4, "performance": 0.3, "execution": 0.3}
            overall_score = (
                coverage_score * weights["coverage"] +
                performance_score * weights["performance"] +
                execution_score * weights["execution"]
            )
            quality_indicators["overall_quality_score"] = overall_score

            # Determine quality status
            if overall_score >= 90:
                quality_indicators["quality_status"] = "excellent"
            elif overall_score >= 80:
                quality_indicators["quality_status"] = "good"
            elif overall_score >= 70:
                quality_indicators["quality_status"] = "acceptable"
            else:
                quality_indicators["quality_status"] = "needs_improvement"

            # Generate actionable insights
            quality_indicators["actionable_insights"] = self._generate_insights()

            # Calculate trend indicators
            quality_indicators["trend_indicators"] = self._calculate_trend_indicators()

        except Exception as e:
            logger.error(f"Error calculating quality indicators: {e}")
            quality_indicators["error"] = str(e)

        self.quality_indicators = quality_indicators
        return quality_indicators

    def _calculate_coverage_score(self) -> float:
        """Calculate coverage component score (0-100)."""
        try:
            coverage_data = self.coverage_metrics.get("overall_coverage", {})
            line_rate = coverage_data.get("line_rate", 0)
            
            # Base score from line coverage
            base_score = min(line_rate, 100)
            
            # Bonus for critical module compliance
            quality_status = self.coverage_metrics.get("quality_status", {})
            if quality_status.get("critical_modules_status") == "pass":
                base_score = min(base_score + 5, 100)
            
            # Penalty for violations
            violations = len(quality_status.get("violations", []))
            penalty = min(violations * 5, 20)
            
            return max(base_score - penalty, 0)

        except Exception as e:
            logger.error(f"Error calculating coverage score: {e}")
            return 0.0

    def _calculate_performance_score(self) -> float:
        """Calculate performance component score (0-100)."""
        try:
            sla_compliance = self.performance_metrics.get("sla_compliance", {})
            
            # Base score from SLA compliance
            base_score = 100
            
            # Deduct for SLA violations
            data_loading_violations = len(sla_compliance.get("data_loading_sla", {}).get("violations", []))
            transform_violations = len(sla_compliance.get("dataframe_transformation_sla", {}).get("violations", []))
            
            total_violations = data_loading_violations + transform_violations
            penalty = min(total_violations * 15, 60)
            
            # Additional penalty for regression
            if self.performance_metrics.get("performance_trends", {}).get("regression_detected", False):
                penalty += 20
            
            return max(base_score - penalty, 0)

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0

    def _calculate_execution_score(self) -> float:
        """Calculate test execution component score (0-100)."""
        try:
            execution_summary = self.test_execution_metrics.get("execution_summary", {})
            pass_rate = execution_summary.get("pass_rate", 0)
            
            # Base score from pass rate
            base_score = pass_rate
            
            # Bonus for comprehensive test coverage
            total_tests = execution_summary.get("total_tests", 0)
            if total_tests >= 100:
                base_score = min(base_score + 5, 100)
            
            # Penalty for excessive execution time
            total_time = execution_summary.get("time", 0)
            if total_time > 300:  # 5 minutes
                penalty = min((total_time - 300) / 60 * 5, 20)
                base_score = max(base_score - penalty, 0)
            
            return base_score

        except Exception as e:
            logger.error(f"Error calculating execution score: {e}")
            return 0.0

    def _generate_insights(self) -> List[str]:
        """Generate actionable insights based on metrics analysis."""
        insights = []

        try:
            # Coverage insights
            coverage_violations = self.coverage_metrics.get("quality_status", {}).get("violations", [])
            if coverage_violations:
                insights.append(f"Coverage violations detected: {len(coverage_violations)} modules below threshold")
                insights.append("Action: Run test suite with coverage analysis and add tests for uncovered code paths")

            # Performance insights
            sla_compliance = self.performance_metrics.get("sla_compliance", {})
            if sla_compliance.get("overall_status") == "fail":
                insights.append("Performance SLA violations detected")
                insights.append("Action: Profile slow operations and optimize data loading/transformation algorithms")

            # Execution insights
            execution_summary = self.test_execution_metrics.get("execution_summary", {})
            if execution_summary.get("pass_rate", 100) < 95:
                insights.append("Test pass rate below 95% - investigate test failures")
                insights.append("Action: Review failed tests and fix underlying issues or update test expectations")

            # Trend insights
            if self.coverage_metrics.get("trend_analysis", {}).get("regression_detected", False):
                insights.append("Coverage regression detected in recent commits")
                insights.append("Action: Review recent changes and ensure adequate test coverage for new code")

            if not insights:
                insights.append("All quality metrics are within acceptable ranges")
                insights.append("Recommendation: Continue maintaining current testing practices")

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append(f"Error analyzing metrics: {e}")

        return insights

    def _calculate_trend_indicators(self) -> Dict[str, Any]:
        """Calculate trend indicators across all metrics."""
        trends = {
            "coverage_trend": "stable",
            "performance_trend": "stable",
            "execution_trend": "stable",
            "overall_direction": "stable"
        }

        try:
            # Coverage trend
            coverage_trend = self.coverage_metrics.get("trend_analysis", {})
            if coverage_trend.get("regression_detected", False):
                trends["coverage_trend"] = "declining"
            elif coverage_trend.get("trend_direction") == "improving":
                trends["coverage_trend"] = "improving"

            # Performance trend
            performance_trend = self.performance_metrics.get("performance_trends", {})
            if performance_trend.get("regression_detected", False):
                trends["performance_trend"] = "declining"

            # Overall trend assessment
            declining_trends = sum(1 for trend in trends.values() if trend == "declining")
            improving_trends = sum(1 for trend in trends.values() if trend == "improving")
            
            if declining_trends > improving_trends:
                trends["overall_direction"] = "declining"
            elif improving_trends > declining_trends:
                trends["overall_direction"] = "improving"

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")

        return trends

    def generate_unified_report(self, output_format: str = "json") -> Dict[str, Any]:
        """
        Generate unified metrics report combining all collected data.
        
        Args:
            output_format: Output format ('json', 'yaml', or 'html')
            
        Returns:
            Dict containing comprehensive metrics report
        """
        logger.info(f"Generating unified report in {output_format} format...")
        
        unified_report = {
            "metadata": {
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0.0",
                "flyrigloader_version": self._get_library_version(),
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            },
            "summary": {
                "overall_quality_score": self.quality_indicators.get("overall_quality_score", 0),
                "quality_status": self.quality_indicators.get("quality_status", "unknown"),
                "total_tests": self.test_execution_metrics.get("execution_summary", {}).get("total_tests", 0),
                "coverage_percentage": self.coverage_metrics.get("overall_coverage", {}).get("line_rate", 0),
                "sla_compliance": self.performance_metrics.get("sla_compliance", {}).get("overall_status", "unknown")
            },
            "detailed_metrics": {
                "coverage": self.coverage_metrics,
                "performance": self.performance_metrics,
                "test_execution": self.test_execution_metrics,
                "quality_indicators": self.quality_indicators
            },
            "recommendations": self.quality_indicators.get("actionable_insights", []),
            "trend_analysis": self.quality_indicators.get("trend_indicators", {})
        }

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "json":
            output_file = self.metrics_output_dir / f"unified_metrics_report_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(unified_report, f, indent=2, default=str)
        
        elif output_format.lower() == "yaml":
            import yaml
            output_file = self.metrics_output_dir / f"unified_metrics_report_{timestamp}.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(unified_report, f, default_flow_style=False)
        
        elif output_format.lower() == "html":
            output_file = self.metrics_output_dir / f"unified_metrics_report_{timestamp}.html"
            self._generate_html_report(unified_report, output_file)

        # Also save as latest report
        latest_file = self.metrics_output_dir / f"latest_metrics_report.{output_format.lower()}"
        if output_format.lower() == "json":
            with open(latest_file, 'w') as f:
                json.dump(unified_report, f, indent=2, default=str)
        elif output_format.lower() == "yaml":
            import yaml
            with open(latest_file, 'w') as f:
                yaml.dump(unified_report, f, default_flow_style=False)
        elif output_format.lower() == "html":
            self._generate_html_report(unified_report, latest_file)

        logger.info(f"Unified report saved to: {output_file}")
        return unified_report

    def _get_library_version(self) -> str:
        """Get flyrigloader library version."""
        try:
            import pkg_resources
            return pkg_resources.get_distribution("flyrigloader").version
        except:
            return "unknown"

    def _generate_html_report(self, report_data: Dict[str, Any], output_file: Path) -> None:
        """Generate HTML report from unified metrics data."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>FlyRigLoader Test Metrics Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .metric-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .success { background-color: #d4edda; border-color: #c3e6cb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .danger { background-color: #f8d7da; border-color: #f5c6cb; }
        .score { font-size: 2em; font-weight: bold; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyRigLoader Test Metrics Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Quality Status: <strong>{quality_status}</strong></p>
    </div>
    
    <div class="metric-card {quality_class}">
        <h2>Overall Quality Score</h2>
        <div class="score">{quality_score:.1f}/100</div>
    </div>
    
    <div class="metric-card">
        <h2>Summary Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{total_tests}</td></tr>
            <tr><td>Coverage Percentage</td><td>{coverage:.1f}%</td></tr>
            <tr><td>SLA Compliance</td><td>{sla_status}</td></tr>
        </table>
    </div>
    
    <div class="metric-card">
        <h2>Recommendations</h2>
        <ul>
        {recommendations}
        </ul>
    </div>
    
    <div class="metric-card">
        <h2>Detailed Metrics</h2>
        <pre>{detailed_json}</pre>
    </div>
</body>
</html>
        """

        # Determine quality class for styling
        quality_score = report_data["summary"]["overall_quality_score"]
        if quality_score >= 90:
            quality_class = "success"
        elif quality_score >= 70:
            quality_class = "warning"
        else:
            quality_class = "danger"

        # Format recommendations
        recommendations_html = "\n".join(
            f"<li>{rec}</li>" for rec in report_data["recommendations"]
        )

        # Generate HTML
        html_content = html_template.format(
            timestamp=report_data["metadata"]["report_timestamp"],
            quality_status=report_data["summary"]["quality_status"],
            quality_score=quality_score,
            quality_class=quality_class,
            total_tests=report_data["summary"]["total_tests"],
            coverage=report_data["summary"]["coverage_percentage"],
            sla_status=report_data["summary"]["sla_compliance"],
            recommendations=recommendations_html,
            detailed_json=json.dumps(report_data["detailed_metrics"], indent=2)
        )

        with open(output_file, 'w') as f:
            f.write(html_content)

    def run_collection_pipeline(
        self,
        output_format: str = "json",
        include_trends: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete metrics collection pipeline.
        
        Args:
            output_format: Output format for reports ('json', 'yaml', 'html')
            include_trends: Whether to include historical trend analysis
            
        Returns:
            Dict containing unified metrics report
        """
        logger.info("Starting comprehensive test metrics collection pipeline...")
        
        try:
            # Set historical analysis flag
            self.include_historical = include_trends
            
            # Collect all metrics
            logger.info("Phase 1/4: Collecting coverage metrics...")
            self.collect_coverage_metrics()
            
            logger.info("Phase 2/4: Collecting performance metrics...")
            self.collect_performance_metrics()
            
            logger.info("Phase 3/4: Collecting test execution metrics...")
            self.collect_test_execution_metrics()
            
            logger.info("Phase 4/4: Calculating quality indicators...")
            self.calculate_quality_indicators()
            
            # Generate unified report
            logger.info("Generating unified metrics report...")
            unified_report = self.generate_unified_report(output_format)
            
            logger.success("Test metrics collection pipeline completed successfully!")
            return unified_report
            
        except Exception as e:
            logger.error(f"Error in metrics collection pipeline: {e}")
            raise


def main():
    """Main entry point for the test metrics collection script."""
    parser = argparse.ArgumentParser(
        description="Collect comprehensive test metrics for flyrigloader testing infrastructure"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for test results (default: current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for metrics (default: tests/coverage/metrics)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "html"],
        default="json",
        help="Output format for reports (default: json)"
    )
    parser.add_argument(
        "--no-trends",
        action="store_true",
        help="Disable historical trend analysis"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    else:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

    try:
        # Initialize collector
        collector = TestMetricsCollector(
            base_dir=args.base_dir,
            metrics_output_dir=args.output_dir,
            include_historical=not args.no_trends
        )

        # Run collection pipeline
        unified_report = collector.run_collection_pipeline(
            output_format=args.format,
            include_trends=not args.no_trends
        )

        # Print summary
        if not args.quiet:
            summary = unified_report["summary"]
            print("\n" + "="*60)
            print("TEST METRICS COLLECTION SUMMARY")
            print("="*60)
            print(f"Overall Quality Score: {summary['overall_quality_score']:.1f}/100")
            print(f"Quality Status: {summary['quality_status']}")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Coverage: {summary['coverage_percentage']:.1f}%")
            print(f"SLA Compliance: {summary['sla_compliance']}")
            print("="*60)

        # Exit with appropriate code
        quality_score = unified_report["summary"]["overall_quality_score"]
        if quality_score >= 70:
            sys.exit(0)
        else:
            logger.error("Quality score below acceptable threshold (70)")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()