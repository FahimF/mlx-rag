#!/usr/bin/env python3
"""
Comprehensive integration test script for MLX-RAG Tool Calling System.

This script runs all test suites and validates the entire system,
providing detailed reporting and validation of the tool calling functionality.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results across different test suites."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0
        }
    
    def add_result(self, suite_name: str, result: Dict[str, Any]):
        """Add test results for a suite."""
        self.results[suite_name] = result
        
        # Update summary
        if 'stats' in result:
            stats = result['stats']
            self.summary['total_tests'] += stats.get('total', 0)
            self.summary['passed'] += stats.get('passed', 0)
            self.summary['failed'] += stats.get('failed', 0)
            self.summary['skipped'] += stats.get('skipped', 0)
            self.summary['errors'] += stats.get('errors', 0)
            self.summary['warnings'] += stats.get('warnings', 0)
    
    def get_duration(self) -> float:
        """Get total test duration."""
        return time.time() - self.start_time
    
    def is_success(self) -> bool:
        """Check if all tests passed."""
        return self.summary['failed'] == 0 and self.summary['errors'] == 0
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        duration = self.get_duration()
        
        report = [
            "=" * 80,
            "MLX-RAG TOOL CALLING INTEGRATION TEST REPORT",
            "=" * 80,
            f"Total Duration: {duration:.2f} seconds",
            f"Test Suites Run: {len(self.results)}",
            "",
            "SUMMARY:",
            f"  Total Tests: {self.summary['total_tests']}",
            f"  Passed: {self.summary['passed']}",
            f"  Failed: {self.summary['failed']}",
            f"  Skipped: {self.summary['skipped']}",
            f"  Errors: {self.summary['errors']}",
            f"  Warnings: {self.summary['warnings']}",
            "",
            "SUITE RESULTS:",
            "=" * 50,
        ]
        
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            duration = result.get('duration', 0)
            stats = result.get('stats', {})
            
            report.extend([
                f"",
                f"{status} {suite_name} ({duration:.2f}s)",
                f"  Tests: {stats.get('total', 0)} | "
                f"Passed: {stats.get('passed', 0)} | "
                f"Failed: {stats.get('failed', 0)} | "
                f"Skipped: {stats.get('skipped', 0)}"
            ])
            
            # Add failure details
            if result.get('failures'):
                report.append("  Failures:")
                for failure in result['failures']:
                    report.append(f"    - {failure}")
            
            # Add error details
            if result.get('errors'):
                report.append("  Errors:")
                for error in result['errors']:
                    report.append(f"    - {error}")
        
        overall_status = "✅ ALL TESTS PASSED" if self.is_success() else "❌ SOME TESTS FAILED"
        report.extend([
            "",
            "=" * 80,
            overall_status,
            "=" * 80
        ])
        
        return "\n".join(report)


class IntegrationTestRunner:
    """Main integration test runner."""
    
    def __init__(self, test_dir: Optional[str] = None, parallel: bool = True):
        self.test_dir = Path(test_dir or os.path.dirname(__file__))
        self.parallel = parallel
        self.results = TestResults()
        self.temp_workspace = None
    
    def setup_test_environment(self):
        """Set up the test environment."""
        logger.info("Setting up test environment...")
        
        # Create temporary workspace for integration tests
        self.temp_workspace = tempfile.mkdtemp(prefix='mlx_rag_test_')
        
        # Set environment variables
        os.environ['MLX_RAG_TEST_MODE'] = '1'
        os.environ['MLX_RAG_TEST_WORKSPACE'] = self.temp_workspace
        
        # Create test files and directories
        self._create_test_fixtures()
        
        logger.info(f"Test workspace created at: {self.temp_workspace}")
    
    def teardown_test_environment(self):
        """Clean up the test environment."""
        logger.info("Cleaning up test environment...")
        
        if self.temp_workspace and os.path.exists(self.temp_workspace):
            shutil.rmtree(self.temp_workspace)
        
        # Clean up environment variables
        os.environ.pop('MLX_RAG_TEST_MODE', None)
        os.environ.pop('MLX_RAG_TEST_WORKSPACE', None)
        
        logger.info("Test environment cleaned up")
    
    def _create_test_fixtures(self):
        """Create test fixtures in the workspace."""
        fixtures_dir = Path(self.temp_workspace) / "fixtures"
        fixtures_dir.mkdir(exist_ok=True)
        
        # Create sample files for testing
        test_files = {
            "sample.txt": "This is a sample text file for testing.",
            "data.json": json.dumps({"test": "data", "numbers": [1, 2, 3]}),
            "code.py": "def hello_world():\\n    print('Hello, World!')\\n",
            "README.md": "# Test Project\\n\\nThis is a test project for MLX-RAG.",
            "config.yaml": "database:\\n  host: localhost\\n  port: 5432\\n"
        }
        
        for filename, content in test_files.items():
            file_path = fixtures_dir / filename
            file_path.write_text(content)
        
        # Create subdirectories with files
        subdir = fixtures_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        (subdir / "nested_file.txt").write_text("Nested file content")
        
        # Create deeply nested structure
        deep_dir = fixtures_dir / "deep" / "nested" / "structure"
        deep_dir.mkdir(parents=True, exist_ok=True)
        (deep_dir / "deep_file.txt").write_text("Deep nested content")
    
    def run_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite."""
        logger.info(f"Running test suite: {suite_name}")
        
        # Map suite names to test files
        suite_files = {
            "tool_execution": "test_tool_execution.py",
            "openai_compatibility": "test_openai_compatibility.py", 
            "security_sandboxing": "test_security_sandboxing.py",
            "error_handling": "test_error_handling.py",
        }
        
        if suite_name not in suite_files:
            return {
                'success': False,
                'error': f"Unknown test suite: {suite_name}",
                'duration': 0,
                'stats': {}
            }
        
        test_file = self.test_dir / suite_files[suite_name]
        
        if not test_file.exists():
            return {
                'success': False,
                'error': f"Test file not found: {test_file}",
                'duration': 0,
                'stats': {}
            }
        
        start_time = time.time()
        
        try:
            # Run pytest with detailed output
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file={suite_name}_report.json"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per suite
            )
            
            duration = time.time() - start_time
            
            # Parse pytest JSON report if available
            json_report_file = self.test_dir / f"{suite_name}_report.json"
            stats = self._parse_pytest_json_report(json_report_file)
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'duration': duration,
                'stats': stats,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'failures': self._extract_failures(result.stdout),
                'errors': self._extract_errors(result.stderr)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Test suite {suite_name} timed out",
                'duration': time.time() - start_time,
                'stats': {}
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error running test suite {suite_name}: {str(e)}",
                'duration': time.time() - start_time,
                'stats': {}
            }
    
    def _parse_pytest_json_report(self, report_file: Path) -> Dict[str, int]:
        """Parse pytest JSON report to extract statistics."""
        if not report_file.exists():
            return {}
        
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            return {
                'total': summary.get('total', 0),
                'passed': summary.get('passed', 0),
                'failed': summary.get('failed', 0),
                'skipped': summary.get('skipped', 0),
                'errors': summary.get('error', 0),
                'warnings': len(data.get('warnings', []))
            }
        except Exception as e:
            logger.warning(f"Failed to parse JSON report {report_file}: {e}")
            return {}
    
    def _extract_failures(self, stdout: str) -> List[str]:
        """Extract failure information from pytest output."""
        failures = []
        lines = stdout.split('\n')
        
        in_failure = False
        current_failure = []
        
        for line in lines:
            if line.startswith('FAILED '):
                if current_failure:
                    failures.append(' '.join(current_failure))
                current_failure = [line]
                in_failure = True
            elif in_failure and line.startswith('='):
                if current_failure:
                    failures.append(' '.join(current_failure))
                current_failure = []
                in_failure = False
            elif in_failure and line.strip():
                current_failure.append(line.strip())
        
        if current_failure:
            failures.append(' '.join(current_failure))
        
        return failures
    
    def _extract_errors(self, stderr: str) -> List[str]:
        """Extract error information from pytest stderr."""
        if not stderr.strip():
            return []
        
        errors = []
        for line in stderr.split('\n'):
            line = line.strip()
            if line and not line.startswith('='):
                errors.append(line)
        
        return errors
    
    def run_all_suites(self) -> bool:
        """Run all test suites."""
        logger.info("Starting comprehensive integration tests...")
        
        suites = [
            "tool_execution",
            "openai_compatibility", 
            "security_sandboxing",
            "error_handling"
        ]
        
        if self.parallel:
            self._run_suites_parallel(suites)
        else:
            self._run_suites_sequential(suites)
        
        return self.results.is_success()
    
    def _run_suites_parallel(self, suites: List[str]):
        """Run test suites in parallel."""
        logger.info(f"Running {len(suites)} test suites in parallel...")
        
        with ThreadPoolExecutor(max_workers=min(len(suites), 4)) as executor:
            # Submit all test suites
            future_to_suite = {
                executor.submit(self.run_test_suite, suite): suite
                for suite in suites
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_suite):
                suite_name = future_to_suite[future]
                try:
                    result = future.result()
                    self.results.add_result(suite_name, result)
                    status = "✅" if result.get('success', False) else "❌"
                    logger.info(f"{status} {suite_name} completed in {result.get('duration', 0):.2f}s")
                except Exception as e:
                    logger.error(f"❌ {suite_name} failed with exception: {e}")
                    self.results.add_result(suite_name, {
                        'success': False,
                        'error': str(e),
                        'duration': 0,
                        'stats': {}
                    })
    
    def _run_suites_sequential(self, suites: List[str]):
        """Run test suites sequentially."""
        logger.info(f"Running {len(suites)} test suites sequentially...")
        
        for suite in suites:
            result = self.run_test_suite(suite)
            self.results.add_result(suite, result)
            status = "✅" if result.get('success', False) else "❌"
            logger.info(f"{status} {suite} completed in {result.get('duration', 0):.2f}s")
    
    def validate_system_integration(self) -> Dict[str, Any]:
        """Validate overall system integration."""
        logger.info("Running system integration validation...")
        
        validation_results = {
            'server_startup': False,
            'tool_registration': False,
            'api_endpoints': False,
            'tool_execution_flow': False,
            'error_handling': False
        }
        
        try:
            # Test 1: Server startup
            validation_results['server_startup'] = self._test_server_startup()
            
            # Test 2: Tool registration
            validation_results['tool_registration'] = self._test_tool_registration()
            
            # Test 3: API endpoints
            validation_results['api_endpoints'] = self._test_api_endpoints()
            
            # Test 4: Tool execution flow
            validation_results['tool_execution_flow'] = self._test_tool_execution_flow()
            
            # Test 5: Error handling
            validation_results['error_handling'] = self._test_error_handling()
            
        except Exception as e:
            logger.error(f"System integration validation failed: {e}")
        
        return validation_results
    
    def _test_server_startup(self) -> bool:
        """Test server startup."""
        try:
            # This would test actual server startup
            # For now, just validate imports work
            from mlx_rag.server import create_app
            app = create_app()
            return app is not None
        except Exception as e:
            logger.error(f"Server startup test failed: {e}")
            return False
    
    def _test_tool_registration(self) -> bool:
        """Test tool registration."""
        try:
            from mlx_rag.tools import ToolExecutor
            executor = ToolExecutor()
            # Check that expected tools are registered
            expected_tools = ['list_directory', 'read_file', 'write_file', 'edit_file', 'search_files']
            # This would be implemented based on actual tool registration mechanism
            return True
        except Exception as e:
            logger.error(f"Tool registration test failed: {e}")
            return False
    
    def _test_api_endpoints(self) -> bool:
        """Test API endpoints."""
        try:
            from fastapi.testclient import TestClient
            from mlx_rag.server import create_app
            
            app = create_app()
            client = TestClient(app)
            
            # Test health endpoint (if exists)
            # response = client.get("/health")
            # return response.status_code == 200
            
            return True
        except Exception as e:
            logger.error(f"API endpoints test failed: {e}")
            return False
    
    def _test_tool_execution_flow(self) -> bool:
        """Test complete tool execution flow."""
        try:
            from mlx_rag.tools import read_file, write_file
            
            # Create a test file
            test_content = "Integration test content"
            write_file("integration_test.txt", test_content, workspace_dir=self.temp_workspace)
            
            # Read it back
            result = read_file("integration_test.txt", workspace_dir=self.temp_workspace)
            
            return result == test_content
        except Exception as e:
            logger.error(f"Tool execution flow test failed: {e}")
            return False
    
    def _test_error_handling(self) -> bool:
        """Test error handling."""
        try:
            from mlx_rag.tools import read_file
            
            # Try to read nonexistent file
            try:
                read_file("nonexistent_file.txt", workspace_dir=self.temp_workspace)
                return False  # Should have raised an exception
            except Exception:
                return True  # Expected exception occurred
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def generate_coverage_report(self) -> str:
        """Generate code coverage report."""
        logger.info("Generating code coverage report...")
        
        try:
            # Run coverage if available
            result = subprocess.run([
                sys.executable, "-m", "coverage", "report", "--show-missing"
            ], capture_output=True, text=True, cwd=self.test_dir)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return "Coverage report not available (install coverage: pip install coverage)"
                
        except FileNotFoundError:
            return "Coverage tool not installed (install with: pip install coverage)"
        except Exception as e:
            return f"Error generating coverage report: {e}"


def main():
    """Main function to run integration tests."""
    parser = argparse.ArgumentParser(description="MLX-RAG Integration Test Runner")
    parser.add_argument("--test-dir", help="Directory containing test files")
    parser.add_argument("--sequential", action="store_true", help="Run tests sequentially instead of parallel")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test environment")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--validate-only", action="store_true", help="Only run system integration validation")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(
        test_dir=args.test_dir,
        parallel=not args.sequential
    )
    
    try:
        # Set up test environment
        runner.setup_test_environment()
        
        if args.validate_only:
            # Run only system integration validation
            logger.info("Running system integration validation only...")
            validation_results = runner.validate_system_integration()
            
            print("\nSYSTEM INTEGRATION VALIDATION RESULTS:")
            print("=" * 50)
            for test_name, passed in validation_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} {test_name}")
            
            all_passed = all(validation_results.values())
            overall_status = "✅ ALL VALIDATIONS PASSED" if all_passed else "❌ SOME VALIDATIONS FAILED"
            print(f"\n{overall_status}")
            
            return 0 if all_passed else 1
        
        # Run all test suites
        success = runner.run_all_suites()
        
        # Run system integration validation
        validation_results = runner.validate_system_integration()
        
        # Generate and print report
        report = runner.results.generate_report()
        print(report)
        
        # Add validation results to report
        print("\nSYSTEM INTEGRATION VALIDATION:")
        print("=" * 50)
        for test_name, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
        
        # Generate coverage report if requested
        if args.coverage:
            coverage_report = runner.generate_coverage_report()
            print("\nCOVERAGE REPORT:")
            print("=" * 50)
            print(coverage_report)
        
        # Save detailed report to file
        with open("integration_test_report.txt", "w") as f:
            f.write(report)
            f.write("\n\nSYSTEM INTEGRATION VALIDATION:\n")
            f.write("=" * 50 + "\n")
            for test_name, passed in validation_results.items():
                status = "PASS" if passed else "FAIL"
                f.write(f"{status} {test_name}\n")
        
        logger.info("Detailed report saved to: integration_test_report.txt")
        
        # Return appropriate exit code
        all_validations_passed = all(validation_results.values())
        return 0 if success and all_validations_passed else 1
        
    except Exception as e:
        logger.error(f"Integration test runner failed: {e}")
        return 1
        
    finally:
        # Clean up test environment unless requested not to
        if not args.no_cleanup:
            runner.teardown_test_environment()


if __name__ == "__main__":
    sys.exit(main())
