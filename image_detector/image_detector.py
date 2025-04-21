from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import torch
import base64
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

app = FastAPI()

# Load the model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # Ensure CPU compatibility
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Convert image to base64 for preview
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Preprocess and predict
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        result_text = "Fake" if prediction == 0 else "Real"

        # Return HTML response with image preview and result
        result_html = f"""
        <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-image:linear-gradient(#004a91, #010207)}}
                    img {{ max-width: 300px; border-radius: 10px; margin-bottom: 20px; }}
                    .result {{ font-size: 20px; font-weight: bold; }}
                    a {{ text-decoration: none; color: white; background: #4CAF50; padding: 10px 20px; border-radius: 5px; }}
                    a:hover {{ background: #45a049; }}
                </style>
            </head>
            <body>
                <h1 style="color: #16f7fa;">Prediction Result</h1>
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image">
                <p style="color: #16f7fa;" class="result">Prediction: {result_text}</p>
                <a href="/">Go Back</a>
            </body>
        </html>
        """
        return HTMLResponse(content=result_html)

    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/")
def main():
    content = """
    <html>
        <head>
            <title >Upload Image</title>
            <style>
           
                # body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px;background-image:linear-gradient(#004a91, #010207) }
               body { 
    font-family: Arial, sans-serif; 
    text-align: center; 
    margin-top: 50px;
    background-image: linear-gradient(#004a91, #010207); 
    height: 100vh;  /* Ensures full page coverage */
    background-repeat: no-repeat;  /* Prevents tiling */
    background-attachment: fixed;  /* Keeps background fixed while scrolling */
    background-size: cover;  /* Ensures it covers the full screen */
}

                form { padding: 20px; border-radius: 10px; display: inline-block; }
                input[type="file"] { margin-bottom: 20px; }
                input[type="submit"] { background: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; border: none; cursor: pointer; }
                input[type="submit"]:hover { background: #45a049; }
                img { max-width: 300px; margin-top: 20px; border-radius: 10px; display: none; }
            </style>
            <script>
                function previewImage(event) {
                    var reader = new FileReader();
                    reader.onload = function(){
                        var img = document.getElementById('preview');
                        img.src = reader.result;
                        img.style.display = 'block';
                    }
                    reader.readAsDataURL(event.target.files[0]);
                }
            </script>
        </head>
        <body>
            <h1 style="color: #16f7fa;">IS YOUR IMAGE FAKE ? CHECK IT !</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" onchange="previewImage(event)">
                <br>
                <img id="preview" alt="Image Preview">
                <br>
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
