import json, os, random, csv

random.seed(42)
os.makedirs("data/policies", exist_ok=True)

# Known escalation patterns (family name -> actions)
VULN = {
    "passrole_lambda": ["iam:PassRole", "lambda:CreateFunction", "lambda:InvokeFunction"],
    "passrole_ec2": ["iam:PassRole", "ec2:RunInstances"],
    "passrole_cfn": ["iam:PassRole", "cloudformation:CreateStack"],
    "passrole_glue": ["iam:PassRole", "glue:CreateDevEndpoint"],
    "create_policy_version": ["iam:CreatePolicyVersion"],
    "set_default_version": ["iam:SetDefaultPolicyVersion"],
    "attach_user_policy": ["iam:AttachUserPolicy"],
    "attach_role_policy": ["iam:AttachRolePolicy"],
    "put_user_policy": ["iam:PutUserPolicy"],
    "put_role_policy": ["iam:PutRolePolicy"],
    "create_access_key": ["iam:CreateAccessKey"],
    "create_login_profile": ["iam:CreateLoginProfile"],
    "update_login_profile": ["iam:UpdateLoginProfile"],
    "update_lambda_code": ["lambda:UpdateFunctionCode"],
    "admin_wildcard": ["*"],
    "iam_wildcard": ["iam:*"],
}

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

def make_policy(actions, resource):
    return {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": actions, "Resource": resource}],
    }

rows = []

# Vulnerable policies: 6 variants per family
for fam, acts in VULN.items():
    for i in range(6):
        extra = random.sample(SAFE_ACTIONS, random.randint(0, 3))
        actions = acts + extra
        random.shuffle(actions)
        resource = random.choice(["*", random.choice(RESOURCES)])
        name = f"{fam}_{i}.json"
        json.dump(make_policy(actions, resource), open(f"data/policies/{name}", "w"), indent=2)
        rows.append((name, 1, fam))

# Safe policies: scoped, least-privilege
for i in range(100):
    actions = random.sample(SAFE_ACTIONS, random.randint(1, 4))
    resource = random.choice(RESOURCES)
    name = f"safe_{i}.json"
    json.dump(make_policy(actions, resource), open(f"data/policies/{name}", "w"), indent=2)
    rows.append((name, 0, f"safe_{i // 5}"))

with open("data/labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "family"])
    w.writerows(rows)

print(f"Wrote {len(rows)} policies")
