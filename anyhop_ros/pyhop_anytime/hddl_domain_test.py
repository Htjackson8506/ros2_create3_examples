import os
from hddl_parser import parse_hddl

prefix = '../../ipc2020-domains/total-order'
for domain in os.listdir(prefix):
    print(f"Testing {domain}...")
    try:
        test_str = open(f"{prefix}/{domain}/domain.hddl").read()
        test_domain = parse_hddl(test_str)
        print("parsed without errors")
    except Exception as e:
        print(f"Exception: {e}")
