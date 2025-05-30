"""
Platform3 Performance Profiling Script
Identifies bottlenecks in adaptive indicators for optimization

This script profiles the adaptive indicators system to identify:
1. CPU-intensive functions
2. Memory usage patterns  
3. I/O bottlenecks
4. Mathematical computation hotspots
"""

import cProfile
import pstats
import io
import time
import tracemalloc
import numpy as np
import pandas as pd
import sys
import os
from memory_profiler import profile
import psutil
import gc

# Add the engines directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

try:
    from engines.ai_enhancement.adaptive_indicators import AdaptiveIndicators
    print("✅ Successfully imported AdaptiveIndicators")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available directories:")
    for root, dirs, files in os.walk('.'):
        if 'adaptive_indicators.py' in files:
            print(f"  Found adaptive_indicators.py in: {root}")

class Platform3Profiler:
    def __init__(self):
        self.results = {}
        self.adaptive_indicators = None
        
    def generate_test_data(self, size=1000):
        """Generate realistic market data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=size, freq='1min')
        
        # Generate correlated price data
        returns = np.random.normal(0, 0.001, size)
        returns[0] = 0
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLC from prices with realistic spreads
        close_prices = prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        high_prices = close_prices + np.random.exponential(0.1, size)
        low_prices = close_prices - np.random.exponential(0.1, size)
        
        volumes = np.random.exponential(1000, size)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        return data
    
    def profile_cpu_usage(self, data):
        """Profile CPU usage of adaptive indicators"""
        print("\n🔍 CPU PROFILING - Adaptive Indicators")
        print("=" * 60)
        
        # Create a string buffer to capture profile output
        pr = cProfile.Profile()
        
        # Profile the execution
        pr.enable()
        try:            # Initialize adaptive indicators
            self.adaptive_indicators = AdaptiveIndicators(
                base_indicator='SMA',
                adaptation_period=50
            )
            
            # Run the adaptive calculation
            result = self.adaptive_indicators.calculate(data)
        except Exception as e:
            print(f"❌ Error during profiling: {e}")
            return None
        finally:
            pr.disable()
        
        # Capture and analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        print(profile_output)
        
        # Save detailed profile to file
        with open('profile_results_detailed.txt', 'w') as f:
            f.write(profile_output)
        
        # Extract key metrics
        ps.sort_stats('tottime')
        top_functions = []
        for func_info in ps.get_stats_profile().func_profiles.values():
            if func_info.cumtime > 0.001:  # Functions taking more than 1ms
                top_functions.append({
                    'function': str(func_info),
                    'total_time': func_info.tottime,
                    'cumulative_time': func_info.cumtime,
                    'calls': func_info.callcount
                })
        
        self.results['cpu_profile'] = {
            'top_functions': sorted(top_functions, key=lambda x: x['total_time'], reverse=True)[:10],
            'profile_output': profile_output
        }
        
        return result
    
    def profile_memory_usage(self, data):
        """Profile memory usage patterns"""
        print("\n💾 MEMORY PROFILING - Adaptive Indicators")
        print("=" * 60)
        
        # Start memory tracing
        tracemalloc.start()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Force garbage collection
            gc.collect()
              # Initialize and run adaptive indicators
            adaptive_indicators = AdaptiveIndicators(
                base_indicator='SMA',
                adaptation_period=50
            )
            
            # Take memory snapshot before execution
            snapshot_before = tracemalloc.take_snapshot()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Execute the indicator calculation
            result = adaptive_indicators.calculate(data)
            
            # Take memory snapshot after execution
            snapshot_after = tracemalloc.take_snapshot()
            memory_after = process.memory_info().rss / 1024 / 1024
            
            # Analyze memory usage
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            print(f"📊 Memory Usage Analysis:")
            print(f"   Initial Memory: {initial_memory:.2f} MB")
            print(f"   Memory Before: {memory_before:.2f} MB")
            print(f"   Memory After: {memory_after:.2f} MB")
            print(f"   Memory Increase: {memory_after - memory_before:.2f} MB")
            print(f"   Memory Growth Rate: {((memory_after - memory_before) / memory_before * 100):.2f}%")
            
            print(f"\n📈 Top Memory Allocations:")
            for index, stat in enumerate(top_stats[:10]):
                print(f"   {index+1}. {stat}")
            
            self.results['memory_profile'] = {
                'initial_memory': initial_memory,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_increase': memory_after - memory_before,
                'growth_rate': (memory_after - memory_before) / memory_before * 100,
                'top_allocations': [str(stat) for stat in top_stats[:10]]
            }
            
        except Exception as e:
            print(f"❌ Memory profiling error: {e}")
        finally:
            tracemalloc.stop()
    
    def profile_execution_time(self, data, iterations=10):
        """Profile execution time with multiple iterations"""
        print(f"\n⏱️ EXECUTION TIME PROFILING - {iterations} iterations")
        print("=" * 60)
          execution_times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                adaptive_indicators = AdaptiveIndicators(
                    base_indicator='SMA',
                    adaptation_period=50
                )
                result = adaptive_indicators.calculate(data)
                
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                execution_times.append(execution_time)
                
                print(f"   Iteration {i+1}: {execution_time:.2f} ms")
                
            except Exception as e:
                print(f"   Iteration {i+1}: ERROR - {e}")
                execution_times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if valid_times:
            avg_time = np.mean(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            std_time = np.std(valid_times)
            
            print(f"\n📊 Execution Time Statistics:")
            print(f"   Average: {avg_time:.2f} ms")
            print(f"   Minimum: {min_time:.2f} ms")
            print(f"   Maximum: {max_time:.2f} ms")
            print(f"   Std Dev: {std_time:.2f} ms")
            print(f"   Target: <10 ms")
            print(f"   Performance Gap: {avg_time - 10:.2f} ms" if avg_time > 10 else "   ✅ Target achieved!")
            
            self.results['execution_time'] = {
                'average': avg_time,
                'minimum': min_time,
                'maximum': max_time,
                'std_dev': std_time,
                'target': 10.0,
                'performance_gap': avg_time - 10.0,
                'all_times': valid_times
            }
        else:
            print("❌ All iterations failed!")
            self.results['execution_time'] = {'error': 'All iterations failed'}
    
    def profile_component_breakdown(self, data):
        """Profile individual components of the adaptive indicators"""
        print("\n🔧 COMPONENT BREAKDOWN PROFILING")
        print("=" * 60)
        
        try:
            adaptive_indicators = AdaptiveIndicators(adaptation_period=50)
            
            # Profile individual components
            components = [
                'kalman_filter',
                'genetic_algorithm', 
                'volatility_scaling',
                'regime_detection',
                'online_learning'
            ]
            
            component_times = {}
            
            for component in components:
                start_time = time.perf_counter()
                
                try:
                    # Test each component individually
                    if hasattr(adaptive_indicators, f'_{component}'):
                        method = getattr(adaptive_indicators, f'_{component}')
                        if callable(method):
                            # Call with test data
                            if component == 'kalman_filter':
                                result = method(data['close'].values)
                            elif component == 'genetic_algorithm':
                                result = method([20], 50)  # parameters, adaptation_period
                            elif component == 'volatility_scaling':
                                result = method(data['close'].values, 20)
                            elif component == 'regime_detection':
                                result = method(data['close'].values)
                            elif component == 'online_learning':
                                result = method(data['close'].values, data['close'].values)
                except Exception as e:
                    print(f"   {component}: ERROR - {e}")
                    continue
                
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                component_times[component] = execution_time
                
                print(f"   {component}: {execution_time:.2f} ms")
            
            # Sort by execution time
            sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 Component Performance Ranking:")
            for i, (component, time_ms) in enumerate(sorted_components):
                percentage = (time_ms / sum(component_times.values())) * 100
                print(f"   {i+1}. {component}: {time_ms:.2f} ms ({percentage:.1f}%)")
            
            self.results['component_breakdown'] = {
                'component_times': component_times,
                'sorted_components': sorted_components
            }
            
        except Exception as e:
            print(f"❌ Component breakdown error: {e}")
    
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations based on profiling results"""
        print("\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Check execution time
        if 'execution_time' in self.results:
            exec_time = self.results['execution_time']
            if exec_time.get('average', 0) > 10:
                gap = exec_time.get('performance_gap', 0)
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Performance',
                    'issue': f"Execution time {exec_time['average']:.2f}ms exceeds 10ms target",
                    'recommendation': f"Optimize core algorithms to reduce by {gap:.2f}ms",
                    'impact': 'Critical for real-time trading'
                })
        
        # Check memory usage
        if 'memory_profile' in self.results:
            mem_profile = self.results['memory_profile']
            if mem_profile.get('memory_increase', 0) > 50:  # More than 50MB increase
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Memory',
                    'issue': f"Memory increase of {mem_profile['memory_increase']:.2f}MB",
                    'recommendation': "Implement object pooling and reduce temporary allocations",
                    'impact': 'Scalability and resource usage'
                })
        
        # Check CPU bottlenecks
        if 'cpu_profile' in self.results:
            cpu_profile = self.results['cpu_profile']
            top_functions = cpu_profile.get('top_functions', [])
            if top_functions:
                slowest = top_functions[0]
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'CPU',
                    'issue': f"CPU bottleneck in top function",
                    'recommendation': "Vectorize operations and optimize mathematical calculations",
                    'impact': 'Overall system performance'
                })
        
        # Check component performance
        if 'component_breakdown' in self.results:
            components = self.results['component_breakdown'].get('sorted_components', [])
            if components:
                slowest_component = components[0]
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Algorithm',
                    'issue': f"Slowest component: {slowest_component[0]} ({slowest_component[1]:.2f}ms)",
                    'recommendation': f"Focus optimization efforts on {slowest_component[0]} algorithm",
                    'impact': 'Targeted performance improvement'
                })
        
        # Print recommendations
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. [{rec['priority']}] {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Impact: {rec['impact']}")
        
        self.results['recommendations'] = recommendations
        
        return recommendations
    
    def save_profiling_report(self):
        """Save comprehensive profiling report"""
        report_path = 'platform3_profiling_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("PLATFORM3 PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Execution time summary
            if 'execution_time' in self.results:
                f.write("EXECUTION TIME ANALYSIS\n")
                f.write("-" * 25 + "\n")
                exec_time = self.results['execution_time']
                f.write(f"Average: {exec_time.get('average', 0):.2f} ms\n")
                f.write(f"Target: 10.00 ms\n")
                f.write(f"Gap: {exec_time.get('performance_gap', 0):.2f} ms\n\n")
            
            # Memory analysis
            if 'memory_profile' in self.results:
                f.write("MEMORY USAGE ANALYSIS\n")
                f.write("-" * 22 + "\n")
                mem_profile = self.results['memory_profile']
                f.write(f"Memory Increase: {mem_profile.get('memory_increase', 0):.2f} MB\n")
                f.write(f"Growth Rate: {mem_profile.get('growth_rate', 0):.2f}%\n\n")
            
            # Recommendations
            if 'recommendations' in self.results:
                f.write("OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(self.results['recommendations']):
                    f.write(f"{i+1}. [{rec['priority']}] {rec['category']}\n")
                    f.write(f"   {rec['recommendation']}\n\n")
        
        print(f"\n📄 Profiling report saved to: {report_path}")
    
    def run_comprehensive_profiling(self, data_size=1000):
        """Run all profiling tests"""
        print("🚀 STARTING COMPREHENSIVE PERFORMANCE PROFILING")
        print("=" * 60)
        print(f"Data size: {data_size} records")
        print(f"Target performance: <10ms execution time")
        print("=" * 60)
        
        # Generate test data
        data = self.generate_test_data(data_size)
        print(f"✅ Generated test data: {len(data)} records")
        
        # Run profiling tests
        self.profile_execution_time(data, iterations=10)
        self.profile_memory_usage(data)
        self.profile_cpu_usage(data)
        self.profile_component_breakdown(data)
        
        # Generate recommendations
        self.generate_optimization_recommendations()
        
        # Save report
        self.save_profiling_report()
        
        print("\n🎉 PROFILING COMPLETE!")
        print("Next steps: Implement optimization recommendations from Phase 1")

if __name__ == "__main__":
    profiler = Platform3Profiler()
    profiler.run_comprehensive_profiling()
