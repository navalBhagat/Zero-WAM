"""Estimate A100 experiment costs; minutes include allocated startup when supplied."""
import argparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',type=float,default=12202+52/60)
    p.add_argument('--episodes',type=int,default=9)
    p.add_argument('--minutes',type=float,default=10)
    p.add_argument('--jobs',type=int,default=9)
    p.add_argument('--startup-minutes',type=float,default=5)
    p.add_argument('--rate',type=float,default=128)
    a=p.parse_args()
    if min(a.budget,a.rate)<=0 or min(a.episodes,a.jobs,a.minutes,a.startup_minutes)<0:
        p.error('Use positive budget/rate and nonnegative counts/time')
    hours=(a.episodes*a.minutes+a.jobs*a.startup_minutes)/60
    cost=hours*a.rate
    print(f'Available: {a.budget:.2f} SBUs = {a.budget/a.rate:.2f} A100 GPU-hours')
    print(f'Estimate: {hours:.2f} GPU-hours = {cost:.2f} SBUs ({cost/a.budget:.1%})')
    print(f'Remaining after estimate: {a.budget-cost:.2f} SBUs')
    print('Planning only. Actual billing uses allocated wall time, including setup and failures.')


if __name__=='__main__': main()
