import logging
import json



class ApiLog:
    """
        A class that represents basic properties for logs and is used to create logs in API.
    """
    def __init__(self, api_name: str, oidc_claim_sub: str,oidc_claim_email: str, oidc_claim_uid: str, project_name: str, developer_email: str) -> None:
        self.api_name = api_name
        self.oidc_claim_sub = oidc_claim_sub
        self.oidc_claim_email = oidc_claim_email
        self.oidc_claim_uid = oidc_claim_uid
        self.project_name = project_name
        self.developer_email = developer_email
        self.logger = logging.getLogger(__name__)

   
    def log_outbound_call(self, provider_name: str, service_name: str, quantity: int = None, forms: str = None, tables: str = None) -> None:
        """
            Creates a log for calls made to external services like AWS, Google & Azure.
            Args:
                provider_name (str):  Name of Service Provider. Can be AWS, GOOGLE or AZURE. 
                service_name  (str): Name of API or service from a service provider. Refer to the comment section
                                     at the end of this file for the possible values. 
                                     Eg. GOOGLE_TRANSLATE_DOCUMENT
                quantity      (int): (Optional) Quantity of pages/characters in the parsed document, image or text
                                     in outbound calls for GOOGLE_TRANSLATE, GOOGLE_TRANSLATE_TEXT_DETECTION, and
                                     Google_Document_Processor.
                forms         (str): (Optional) Used in AWS Textract outbound calls to indicate whether the parsed
                                     document contains forms. Valid values are 'true' or 'false'.
                tables        (str): (Optional) Used in AWS Textract outbound calls to indicate whether the parsed
                                     document contains tables. Valid values are 'true' or 'false'.
        """
        if(quantity is not None):
            new_log = {'API_NAME' : self.api_name , 'CALL_TYPE' : 'outbound', 'SERVICE_NAME' : service_name, \
                    'PROVIDER' : provider_name, 'QUANTITY' : quantity, 'OIDC_CLAIM_sub' : self.oidc_claim_sub, 'PROJECT': self.project_name, \
                    'DEVELOPER_email': self.developer_email}
        elif((forms is not None) and (tables is not None)):
            new_log = {'API_NAME' : self.api_name , 'CALL_TYPE' : 'outbound', 'SERVICE_NAME' : service_name, \
                    'PARAMS': {'Forms': forms, 'Tables': tables},'PROVIDER' : provider_name, 'OIDC_CLAIM_sub' : self.oidc_claim_sub, 'PROJECT': self.project_name, \
                    'DEVELOPER_email': self.developer_email}
        else:
            new_log = {'API_NAME' : self.api_name , 'CALL_TYPE' : 'outbound', 'SERVICE_NAME' : service_name, \
                    'PROVIDER' : provider_name, 'OIDC_CLAIM_sub' : self.oidc_claim_sub, 'PROJECT': self.project_name, \
                    'DEVELOPER_email': self.developer_email}
        log_json = json.dumps(new_log)
        self.logger.info(log_json)

    def log_inbound_call(self) -> None:
        """
            Creates a log when a request is made to the API.
        """
        new_log = {'API_NAME' : self.api_name , 'CALL_TYPE' : 'inbound', \
            'OIDC_CLAIM_sub' : self.oidc_claim_sub, 'OIDC_CLAIM_email' : self.oidc_claim_email,\
            'OIDC_CLAIM_uid' : self.oidc_claim_uid, 'PROJECT': self.project_name, \
            'DEVELOPER_email': self.developer_email}
        log_json = json.dumps(new_log)
        self.logger.info(log_json)




"""
Service Names for Outbound Calls
--------------------------------
AWS
    - AWS_TEXTRACT_ANALYZE_DOCUMENT

GOOGLE
    - GOOGLE_TRANSLATE_DETECT_LANGUAGE
    - GOOGLE_VISION_TEXT_DETECTION
    - GOOGLE_TRANSLATE_DOCUMENT
    - GOOGLE_DOCUMENT_PROCESSOR

AZURE
    - "AZURE_FORM_RECOGNIZER"
"""