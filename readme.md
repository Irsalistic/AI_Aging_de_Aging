# Face Aging API

A FastAPI-based endpoint that provides age transformation capabilities for facial images. The API can age or rejuvenate faces while maintaining natural features and personal characteristics.

## Features

- Age transformation in both directions (younger/older)
- Gender-specific aging patterns
- Configurable transformation strength
- Maintains original facial features and characteristics
- Supports various image formats
- Built-in image preprocessing and resizing

## API Endpoint

### POST /aging

Transforms the age appearance of a face in the provided image.

#### Request Parameters

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| key       | string | Yes      | API authentication key |
| image     | file   | Yes      | Input image file |
| gender    | string | Yes      | Gender of the person ("male" or "female") |
| strength  | float  | No       | Transformation strength (default: 0.6) |

#### Strength Values
- 0.1-0.2: Young transformation (approximately 20 years old)
- Default/Other values: Elderly transformation (approximately 70 years old)

#### Authentication

The API requires a valid authentication key. Contact the system administrator to obtain access credentials.

#### Response

Returns a PNG image with the age-transformed face.

- Content-Type: image/png
- Status Code: 200 (Success)

#### Error Responses

- 401 Unauthorized: Invalid or missing API key
- Other standard HTTP error codes may apply

## Technical Details

- Uses multiple AI models for realistic transformations:
  - RealisticVision v6.0
  - ControlNet v1.1 with scribble and IP2P modules
- Implements face masking for targeted transformations
- Preserves original image dimensions
- Applies intelligent strength adjustments based on the desired outcome

## Usage Example

```python
import requests

url = "http://your-api-endpoint/aging"
files = {
    'image': open('face.png', 'rb')
}
data = {
    'key': 'your-api-key',
    'gender': 'male',
    'strength': 0.6
}

response = requests.post(url, files=files, data=data)
with open('aged_face.png', 'wb') as f:
    f.write(response.content)
```

## Notes

- The API uses specific prompts for age transformation that maintain natural-looking results
- Negative prompts are implemented to avoid unrealistic features
- The transformation process includes multiple steps to ensure quality results
- Image preprocessing is automatically applied to optimize input for the AI models

## Dependencies

- FastAPI
- Pillow (PIL)
- NumPy
- Base64
- BytesIO

## Important Considerations

- Input images should contain clear, front-facing portraits
- Optimal results are achieved with high-quality input images
- Processing time may vary based on image size and server load
- The API maintains original skin tone and key facial features while applying age-related changes
