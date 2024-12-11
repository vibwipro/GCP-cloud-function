#*************************************************************************#
#  Script : funct.py
#  Developed by : Vibhor
#  Developed Date : 2022-08-08
#  Update Date : 2022-08-08
#  Version No : 1.00.01
#*************************************************************************#
# Version No : 1.00.01 : Initial version.
#*************************************************************************#

#****************import python lib*****************************************#
from google.cloud import secretmanager

#****************define python function***********************************#
def fetchSecret():
	client = secretmanager.SecretManagerServiceClient()
	name = "projects/2xxxxxxxxxxxxx2/secrets/abcd/versions/latest"
	response = client.access_secret_version(name=name)
	payload = response.payload.data.decode("UTF-8")
	return payload