import os, os.path
from hddl_parser import parse_hddl

prefix = '../../ipc2020-domains/total-order'
print(os.getcwd())
for domain in os.listdir(prefix):
    print(f"Testing {domain}...")
    domain_dir = f"{prefix}/{domain}"
    if os.path.isdir(domain_dir):
        for problem in os.listdir(domain_dir):
            if problem.endswith("hddl"):
                try:
                    test_str = open(f"{prefix}/{domain}/{problem}").read()
                    test_domain = parse_hddl(test_str)
                    print(f"{problem} parsed without errors")
                except Exception as e:
                    print(f"{domain} {problem} Exception: {e}")
