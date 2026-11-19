"""Print rendering, action and inference costs from episode profiler JSON files."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='Run directories or profile JSON files')
    args = parser.parse_args()
    files = sorted({file for path in args.paths for file in
                    (path.rglob('profile_episode_*.json') if path.is_dir() else [path])})
    if not files:
        parser.error('No profile_episode_*.json files found')
    for path in files:
        data = json.loads(path.read_text())
        print(f'\n{path}\nSuccess: {data["success"]}; wall: {data["wall_seconds"]:.2f}s')
        print(f'{"Category":32} {"Calls":>7} {"Exclusive s":>12} {"Inclusive s":>12}')
        for label, row in sorted(data['timings'].items(), key=lambda item: -item[1]['exclusive_seconds']):
            print(f'{label:32} {row["calls"]:7d} {row["exclusive_seconds"]:12.3f} '
                  f'{row["inclusive_seconds"]:12.3f}')
        missing = {key: value for key, value in data['hooks'].items() if value != 'enabled'}
        if missing:
            print('Uninstrumented hooks:', json.dumps(missing))
        print('Inclusive times overlap. Action remainder is not necessarily pure physics.')


if __name__ == '__main__':
    main()
