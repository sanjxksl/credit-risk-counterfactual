import boto3, json

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

application = {
    "loan_amount": 280000, "term": 360, "property_value": 350000,
    "income": 90000, "credit_score": 620, "ltv": 80.0, "dtir1": 42.0,
    "year": 2019, "loan_limit": "cf", "gender": "Male",
    "approv_in_adv": "pre", "loan_type": "type1", "loan_purpose": "p1",
    "credit_worthiness": "l1", "open_credit": "nopc",
    "business_or_commercial": "nob/c", "neg_ammortization": "not_neg",
    "interest_only": "not_int", "lump_sum_payment": "not_lpsm",
    "occupancy_type": "pr", "total_units": "1U", "age": "35-44",
    "submission_of_application": "to_inst", "region": "central"
}

response = client.invoke_endpoint(
    EndpointName='credit-risk-endpoint',
    ContentType='application/json',
    Body=json.dumps(application)
)
print(json.loads(response['Body'].read()))
