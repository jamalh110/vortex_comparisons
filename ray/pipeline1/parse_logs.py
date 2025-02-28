import os
import glob
from datetime import datetime
import statistics
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_files(directory, pattern="*"):
    """
    Parse log files in the given directory that match the pattern.
    
    Args:
        directory (str): Path to directory containing log files
        pattern (str): Glob pattern for matching files (default: "*")
    
    Returns:
        dict: A dictionary with request IDs as keys and dictionaries of events and timestamps as values
    """
    # Dictionary to store parsed data
    request_data = {}
    total_lines_processed = 0
    successful_lines_parsed = 0
    duplicate_count = 0
    
    # Create full path pattern
    full_pattern = os.path.join(directory, pattern)
    file_paths = glob.glob(full_pattern)
    
    if not file_paths:
        logging.warning(f"No files found matching pattern {full_pattern}")
        return request_data
    
    logging.info(f"Found {len(file_paths)} files matching pattern {full_pattern}")
    
    # Process each file
    for file_path in file_paths:
        if os.path.isdir(file_path):
            continue
        
        try:
            with open(file_path, 'r') as file:
                logging.info(f"Processing file: {file_path}")
                for line_num, line in enumerate(file, 1):
                    total_lines_processed += 1
                    
                    try:
                        parsed_data = parse_log_line(line)
                        if parsed_data:
                            successful_lines_parsed += 1
                            timestamp, log_level, event, request_id = parsed_data
                            
                            # Initialize request_id entry if it doesn't exist
                            if request_id not in request_data:
                                request_data[request_id] = {}
                            
                            # Check for duplicate events
                            if event in request_data[request_id]:
                                duplicate_count += 1
                                logging.warning(f"Duplicate event {event} for request ID {request_id} in {file_path}:{line_num}")
                            
                            # Store event and timestamp
                            request_data[request_id][event] = timestamp
                    except Exception as e:
                        logging.error(f"Error parsing line {line_num} in {file_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
    
    logging.info(f"Total lines processed: {total_lines_processed}")
    logging.info(f"Successfully parsed lines: {successful_lines_parsed}")
    logging.info(f"Processed {len(request_data)} unique request IDs")
    logging.info(f"Found {duplicate_count} duplicate events")
    
    return request_data

def parse_log_line(line):
    """
    Parse a single log line and extract timestamp, log level, event, and request ID.
    
    Args:
        line (str): Log line to parse
    
    Returns:
        tuple: (timestamp, log_level, event, request_id) or None if parsing fails
    """
    parts = line.strip().split(' - ')
    if len(parts) != 3:
        return None
    
    timestamp_str = parts[0]
    log_level = parts[1]
    event_and_id = parts[2].split()
    
    if len(event_and_id) != 2:
        return None
    
    event = event_and_id[0]
    request_id = event_and_id[1]
    
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        return timestamp, log_level, event, request_id
    except ValueError:
        return None

def calculate_average_time_between_events(request_data, event1, event2):
    """
    Calculate the average time between two specified events across all request IDs.
    
    Args:
        request_data (dict): Parsed log data
        event1 (str): First event name
        event2 (str): Second event name
    
    Returns:
        float: Average time difference in seconds or None if no matching pairs found
    """
    time_diffs_with_stats = calculate_time_diffs_with_stats(request_data, event1, event2)
    
    if time_diffs_with_stats and 'mean' in time_diffs_with_stats:
        return time_diffs_with_stats['mean']
    
    return None

def calculate_time_diffs_with_stats(request_data, event1, event2):
    """
    Calculate time differences between two events and return statistics.
    
    Args:
        request_data (dict): Parsed log data
        event1 (str): First event name
        event2 (str): Second event name
    
    Returns:
        dict: Dictionary with statistics and raw time differences
    """
    time_differences = []
    request_ids = []
    
    for request_id, events in request_data.items():
        if event1 in events and event2 in events:
            # Calculate time difference in seconds
            time_diff = (events[event2] - events[event1]).total_seconds()
            time_differences.append(time_diff)
            request_ids.append(request_id)
    
    if not time_differences:
        return {}
    
    result = {
        'time_differences': time_differences,
        'request_ids': request_ids,
        'count': len(time_differences),
        'min': min(time_differences),
        'max': max(time_differences),
        'mean': statistics.mean(time_differences),
        'median': statistics.median(time_differences)
    }
    
    # Standard deviation requires at least 2 samples
    if len(time_differences) > 1:
        result['stdev'] = statistics.stdev(time_differences)
    
    return result

def list_available_events(request_data):
    """
    List all unique event types from the parsed data.
    
    Args:
        request_data (dict): Parsed log data
    
    Returns:
        list: List of unique event types
    """
    events = set()
    for request_events in request_data.values():
        events.update(request_events.keys())
    return sorted(list(events))

def analyze_multiple_event_pairs(request_data, event_pairs):
    """
    Analyze multiple pairs of events and return statistics for each pair.
    
    Args:
        request_data (dict): Parsed log data
        event_pairs (list): List of tuples of event pairs to analyze [(event1, event2), ...]
    
    Returns:
        dict: Dictionary with event pairs as keys and statistics as values
    """
    results = {}
    for event1, event2 in event_pairs:
        pair_key = f"{event1} → {event2}"
        results[pair_key] = calculate_time_diffs_with_stats(request_data, event1, event2)
    
    return results

class LogAnalyzer:
    """A class for analyzing log files and calculating time between events."""
    
    def __init__(self, log_directory=None, file_pattern="*"):
        """
        Initialize LogAnalyzer.
        
        Args:
            log_directory (str, optional): Directory containing log files
            file_pattern (str, optional): Glob pattern for matching files (default: "*")
        """
        self.request_data = {}
        if log_directory:
            self.load_logs(log_directory, file_pattern)
    
    def load_logs(self, directory, pattern="*"):
        """
        Load and parse log files from a directory.
        
        Args:
            directory (str): Directory containing log files
            pattern (str, optional): Glob pattern for matching files (default: "*")
        
        Returns:
            self: For method chaining
        """
        self.request_data = parse_log_files(directory, pattern)
        return self
    
    def get_average_time_between_events(self, event1, event2):
        """
        Calculate the average time between two events.
        
        Args:
            event1 (str): First event name
            event2 (str): Second event name
        
        Returns:
            float: Average time in seconds or None if no matching pairs found
        """
        return calculate_average_time_between_events(self.request_data, event1, event2)
    
    def get_time_stats_between_events(self, event1, event2):
        """
        Get detailed time statistics between two events.
        
        Args:
            event1 (str): First event name
            event2 (str): Second event name
        
        Returns:
            dict: Dictionary with time statistics
        """
        return calculate_time_diffs_with_stats(self.request_data, event1, event2)
    
    def list_events(self):
        """
        List all unique event types in the parsed data.
        
        Returns:
            list: List of event types
        """
        return list_available_events(self.request_data)
    
    def analyze_event_pairs(self, event_pairs):
        """
        Analyze multiple pairs of events and get statistics for all of them.
        
        Args:
            event_pairs (list): List of tuples containing event pairs to analyze [(event1, event2), ...]
            
        Returns:
            dict: Dictionary with event pairs as keys and their statistics as values
        """
        return analyze_multiple_event_pairs(self.request_data, event_pairs)
def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Calculate average time between log events')
    parser.add_argument('directory', help='Directory containing log files')
    parser.add_argument('--pattern', default='*', help='Glob pattern for matching files (default: *)')
    parser.add_argument('--event1', help='First event name')
    parser.add_argument('--event2', help='Second event name')
    parser.add_argument('--list-events', action='store_true', help='List all available events')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    analyzer = LogAnalyzer(args.directory, args.pattern)
    
    if args.list_events:
        events = analyzer.list_events()
        print("Available events:")
        for event in events:
            print(f"  {event}")
        return
    
    if not args.event1 or not args.event2:
        print("Please specify both --event1 and --event2, or use --list-events to see available events")
        return
    
    if args.stats:
        stats = analyzer.get_time_stats_between_events(args.event1, args.event2)
        if stats:
            print(f"Time statistics between {args.event1} and {args.event2}:")
            print(f"  Count: {stats['count']} request IDs")
            print(f"  Min: {stats['min']:.4f} seconds")
            print(f"  Max: {stats['max']:.4f} seconds")
            print(f"  Mean: {stats['mean']:.4f} seconds")
            print(f"  Median: {stats['median']:.4f} seconds")
            if 'stdev' in stats:
                print(f"  Standard Deviation: {stats['stdev']:.4f} seconds")
        else:
            print(f"No matching event pairs found for {args.event1} and {args.event2}")
    else:
        avg_time = analyzer.get_average_time_between_events(args.event1, args.event2)
        if avg_time is not None:
            print(f"Average time between {args.event1} and {args.event2}: {avg_time:.4f} seconds")
        else:
            print(f"No matching event pairs found for {args.event1} and {args.event2}")

if __name__ == "__main__":
    #main()
    analyzer = LogAnalyzer('/users/jamalh11/raylogs')
    events = analyzer.list_events()
    event_pairs = [
    ('Client_Send', 'Client_Rec'),
    ('Ingress_Enter', 'Ingress_Exit'),
    ('StepA_Enter', 'StepA_Exit'),
    ('StepB_Enter', 'StepB_Exit'),
    ('StepC_Enter', 'StepC_Exit'),
    ('StepD_Enter', 'StepD_Exit'),
    ('StepE_Enter', 'StepE_Exit'),
    ('Client_Send', 'Ingress_Enter'),
    ('StepB_Exit', "StepC_Enter"),
    ('StepA_Exit', "StepD_Enter"),
    ('StepC_Exit', "StepD_Enter"),
    ('StepD_Exit', "StepE_Enter"),
    ]
    results = analyzer.analyze_event_pairs(event_pairs)
    #print(results)
    print("\n\n\n\n")

    #print the key, count, min, max, mean, median, and stdev for each key in the results dictionary
    for key, value in results.items():
        print(f"Event Pair: {key}")
        print(f"  Count: {value['count']}")
        print(f"  Min: {value['min']*1000:.1f} ms")
        print(f"  Max: {value['max']*1000:.1f} ms")
        print(f"  Mean: {value['mean']*1000:.1f} ms")
        print(f"  Median: {value['median']*1000:.1f} ms")
        if 'stdev' in value:
            print(f"  Standard Deviation: {value['stdev']*1000:.1f} ms")
        print("\n\n\n")

