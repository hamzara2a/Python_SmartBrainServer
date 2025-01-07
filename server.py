from flask import Flask
from flask import request
from flask_cors import CORS
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<h1>This is my personal server for the SmartBrain project!</h1>"

@app.route("/predict", methods=["GET", "POST"])
def make_pred():
    incomingReq = request.get_json()
    PAT = "YOUR_PAT_HERE"
    USER_ID = 'hamza_pr1vate'
    APP_ID = 'TestingAgain'
    MODEL_ID = 'face-detection'
    MODEL_VERSION_ID = '6dc7e46bc9124c5c8824be4822abe105'
    IMAGE_URL = incomingReq["input"]

    
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            url=IMAGE_URL
                        # base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
        
    regions = post_model_outputs_response.outputs[0].data.regions
    regionArray = []
    
    for region in regions:
        regionBox = {}
        # Accessing and rounding the bounding box values
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)
        regionBox["top_row"] = top_row
        regionBox["left_col"] = left_col
        regionBox["bottom_row"] = bottom_row
        regionBox["right_col"] = right_col
        regionArray.append(regionBox)
    
    return regionArray

if __name__ == "__main__":
    app.run()