import json, os, random, csv

random.seed(7)
OUT = "data/policies_v2"
os.makedirs(OUT, exist_ok=True)

SAFE_ACTIONS = [
    "s3:GetObject", "s3:ListBucket", "s3:PutObject",
    "logs:CreateLogStream", "logs:PutLogEvents",
    "dynamodb:GetItem", "dynamodb:Query", "dynamodb:PutItem",
    "sqs:SendMessage", "sqs:ReceiveMessage",
    "cloudwatch:PutMetricData", "ec2:DescribeInstances",
    "sns:Publish", "kms:Decrypt",
]
RESOURCES = [
    "arn:aws:s3:::app-data/*", "arn:aws:s3:::reports-bucket/*",
    "arn:aws:dynamodb:ap-south-1:123456789012:table/orders",
    "arn:aws:sqs:ap-south-1:123456789012:jobs",
    "arn:aws:logs:ap-south-1:123456789012:*",
]
ROLE_ARNS = [
    "arn:aws:iam::123456789012:role/lambda-basic-exec",
    "arn:aws:iam::123456789012:role/etl-readonly",
    "arn:aws:iam::123456789012:role/batch-worker",
]
COMPUTE = {
    "lambda": "lambda:CreateFunction",
    "ec2": "ec2:RunInstances",
    "cfn": "cloudformation:CreateStack",
    "glue": "glue:CreateDevEndpoint",
}
SINGLE = [
    "iam:CreatePolicyVersion", "iam:AttachUserPolicy", "iam:PutRolePolicy",
    "iam:CreateAccessKey", "iam:UpdateLoginProfile", "lambda:UpdateFunctionCode",
]


def stmt(actions, resource):
    return {"Effect": "Allow", "Action": actions, "Resource": resource}


def policy(*stmts):
    return {"Version": "2012-10-17", "Statement": list(stmts)}


def noise():
    """Optional harmless extra statement."""
    if random.random() < 0.6:
        return [stmt(random.sample(SAFE_ACTIONS, random.randint(1, 3)),
                     random.choice(RESOURCES))]
    return []


rows = []
counter = {}


def save(p, label, family):
    counter[family] = counter.get(family, 0) + 1
    name = f"{family}_{counter[family]}.json"
    json.dump(p, open(os.path.join(OUT, name), "w"), indent=2)
    rows.append((name, label, family))


# ---------- VULNERABLE ----------
# PassRole on * combined with a compute-creating action
for svc, act in COMPUTE.items():
    for _ in range(15):
        if random.random() < 0.5:
            p = policy(stmt(["iam:PassRole", act], "*"), *noise())
        else:
            p = policy(stmt("iam:PassRole", "*"), stmt(act, "*"), *noise())
        save(p, 1, f"vuln_passrole_{svc}")

# Standalone escalation actions on *
for act in SINGLE:
    for _ in range(8):
        save(policy(stmt(act, "*"), *noise()), 1, f"vuln_single_{act.split(':')[1].lower()}")

# Wildcards
for fam, acts in [("vuln_iam_wildcard", ["iam:*"]), ("vuln_admin", ["*"])]:
    for _ in range(8):
        save(policy(stmt(acts, "*"), *noise()), 1, fam)

# ---------- SAFE LOOK-ALIKES ----------
# PassRole alone is not an escalation
for _ in range(20):
    save(policy(stmt("iam:PassRole", random.choice(["*"] + ROLE_ARNS)), *noise()),
         0, "safe_passrole_alone")

# Compute action alone is not an escalation
for svc, act in COMPUTE.items():
    for _ in range(6):
        save(policy(stmt(act, "*"), *noise()), 0, f"safe_compute_alone_{svc}")

# PassRole scoped to ONE specific role + compute action.
# Assumption: that role is low-privilege, so no escalation.
for svc, act in COMPUTE.items():
    for _ in range(8):
        save(policy(stmt("iam:PassRole", random.choice(ROLE_ARNS)), stmt(act, "*"), *noise()),
             0, f"safe_scoped_passrole_{svc}")

# Ordinary least-privilege policies
for i in range(80):
    p = policy(stmt(random.sample(SAFE_ACTIONS, random.randint(1, 4)),
                    random.choice(RESOURCES)))
    save(p, 0, f"safe_ordinary_{i // 5}")

with open("data/labels_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "family"])
    w.writerows(rows)

print(f"Wrote {len(rows)} policies  ({sum(r[1] for r in rows)} vulnerable)")
