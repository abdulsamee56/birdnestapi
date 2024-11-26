from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .model_handler import predict
import os

class BirdPredictionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # To handle image uploads

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        temp_image_path = f"temp_{image_file.name}"

        try:
            # Save the uploaded image temporarily
            with open(temp_image_path, 'wb+') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)

            # Run prediction
            prediction = predict(temp_image_path)

            # Remove the temporary image
            os.remove(temp_image_path)

            # If prediction is None, handle the error gracefully
            if not prediction:
                return Response({"error": "Prediction failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Extract bird name and confidence
            bird_name = prediction["label"].split(".")[1].replace("_", " ")
            confidence = prediction["confidence"]

            # Return predictions as a JSON response
            return Response(
                {"bird_name": bird_name, "confidence": confidence},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
