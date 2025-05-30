"""
Platform3 Performance Profiling Script - Fixed Version
Identifies bottlenecks in adaptive indicators for optimization
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
import psutil
import gc

# Add the engines directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

try:
    from engines.ai_enhancement.adaptive_indicators import AdaptiveIndicators
    print("✅ Successfully imported AdaptiveIndicators")
except ImportError as e:
    print(f"❌ Import error: {e}")

class Platform3Profiler:
    def __init__(self):
        self.results = {}
        
    def generate_test_data(self, size=1000):
        """Generate realistic market data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='1min')
        
        returns = np.random.normal(0, 0.001, size)
        returns[0] = 0
        prices = 100 * np.exp(np.cumsum(returns))
        
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
                execution_time = (end_time - start_time) * 1000
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
    
    def profile_memory_usage(self, data):
        """Profile memory usage patterns"""
        print("\n💾 MEMORY PROFILING - Adaptive Indicators")
        print("=" * 60)
        
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            gc.collect()
            
            adaptive_indicators = AdaptiveIndicators(
                base_indicator='SMA',
                adaptation_period=50
            )
            
            snapshot_before = tracemalloc.take_snapshot()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            result = adaptive_indicators.calculate(data)
            
            snapshot_after = tracemalloc.take_snapshot()
            memory_after = process.memory_info().rss / 1024 / 1024
            
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            print(f"📊 Memory Usage Analysis:")
            print(f"   Initial Memory: {initial_memory:.2f} MB")
            print(f"   Memory Before: {memory_before:.2f} MB")
            print(f"   Memory After: {memory_after:.2f} MB")
            print(f"   Memory Increase: {memory_after - memory_before:.2f} MB")
            
            if memory_before > 0:
                growth_rate = ((memory_after - memory_before) / memory_before * 100)
                print(f"   Memory Growth Rate: {growth_rate:.2f}%")
            
            print(f"\n📈 Top Memory Allocations:")
            for index, stat in enumerate(top_stats[:5]):
                print(f"   {index+1}. {stat}")
            
            self.results['memory_profile'] = {
                'initial_memory': initial_memory,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_increase': memory_after - memory_before,
                'top_allocations': [str(stat) for stat in top_stats[:5]]
            }
            
        except Exception as e:
            print(f"❌ Memory profiling error: {e}")
        finally:
            tracemalloc.stop()
    
    def profile_cpu_usage(self, data):
        """Profile CPU usage of adaptive indicators"""
        print("\n🔍 CPU PROFILING - Adaptive Indicators")
        print("=" * 60)
        
        pr = cProfile.Profile()
        
        pr.enable()
        try:
            adaptive_indicators = AdaptiveIndicators(
                base_indicator='SMA',
                adaptation_period=50
            )
            result = adaptive_indicators.calculate(data)
        except Exception as e:
            print(f"❌ Error during profiling: {e}")
            return None
        finally:
            pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)
        
        profile_output = s.getvalue()
        print(profile_output)
        
        with open('profile_results_detailed.txt', 'w') as f:
            f.write(profile_output)
        
        self.results['cpu_profile'] = {'profile_output': profile_output}
        
        return result
    
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations"""
        print("\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        if 'execution_time' in self.results:
            exec_time = self.results['execution_time']
            if 'average' in exec_time and exec_time['average'] > 10:
                gap = exec_time.get('performance_gap', 0)
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Performance',
                    'issue': f"Execution time {exec_time['average']:.2f}ms exceeds 10ms target",
                    'recommendation': f"Optimize core algorithms to reduce by {gap:.2f}ms",
                    'impact': 'Critical for real-time trading'
                })
        
        if 'memory_profile' in self.results:
            mem_profile = self.results['memory_profile']
            if mem_profile.get('memory_increase', 0) > 50:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Memory',
                    'issue': f"Memory increase of {mem_profile['memory_increase']:.2f}MB",
                    'recommendation': "Implement object pooling and reduce temporary allocations",
                    'impact': 'Scalability and resource usage'
                })
        
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
            
            if 'execution_time' in self.results:
                f.write("EXECUTION TIME ANALYSIS\n")
                f.write("-" * 25 + "\n")
                exec_time = self.results['execution_time']
                if 'average' in exec_time:
                    f.write(f"Average: {exec_time['average']:.2f} ms\n")
                    f.write(f"Target: 10.00 ms\n")
                    f.write(f"Gap: {exec_time.get('performance_gap', 0):.2f} ms\n\n")
            
            if 'memory_profile' in self.results:
                f.write("MEMORY USAGE ANALYSIS\n")
                f.write("-" * 22 + "\n")
                mem_profile = self.results['memory_profile']
                f.write(f"Memory Increase: {mem_profile.get('memory_increase', 0):.2f} MB\n\n")
            
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
        
        data = self.generate_test_data(data_size)
        print(f"✅ Generated test data: {len(data)} records")
        
        self.profile_execution_time(data, iterations=5)
        self.profile_memory_usage(data)
        self.profile_cpu_usage(data)
        
        self.generate_optimization_recommendations()
        self.save_profiling_report()
        
        print("\n🎉 PROFILING COMPLETE!")
        print("Check the detailed profiling report for optimization targets.")

if __name__ == "__main__":
    profiler = Platform3Profiler()
    profiler.run_comprehensive_profiling()
