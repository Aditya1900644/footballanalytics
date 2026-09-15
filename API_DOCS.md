# API Documentation for Football Analytics Backend

This document outlines the API endpoints for the Automated Football Match Video Analytics System backend, implemented using FastAPI.

## 1. Base URL and CORS Setup

The backend server typically runs on `http://localhost:8000` when started locally. All endpoints described below are relative to this base URL.

### Cross-Origin Resource Sharing (CORS)

CORS is configured to allow requests from **any origin (`*`)** for development and demonstration purposes. This means any frontend application, regardless of its domain, can interact with this API. All HTTP methods, credentials, and headers are allowed.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 2. Endpoint: `POST /upload`

Initiates the video analysis process by uploading a video file and optional club colors. The analysis runs as a background task, and a `job_id` is returned for tracking progress.

- **URL**: `/upload`
- **Method**: `POST`
- **Description**: Uploads a football match video, triggers a background analysis task, and returns a job identifier.

### Parameters (Query or Form Data)

| Name | Type   | Description                                                                  | Default       |
| :--- | :----- | :--------------------------------------------------------------------------- | :------------ |
| `file` | `UploadFile` | The video file to be analyzed. Must be sent as `multipart/form-data`.        | (Required)    |
| `c1`   | `str`  | Hexadecimal color code for Club 1's jersey (e.g., `#E8F7F8`).              | `#E8F7F8`     |
| `c2`   | `str`  | Hexadecimal color code for Club 2's jersey (e.g., `#ACFB91`).              | `#ACFB91`      |

### Request Payload Example (multipart/form-data)

```http
POST /upload HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="my_match.mp4"
Content-Type: video/mp4

<binary video data>
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="c1"

#FF0000
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="c2"

#0000FF
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

### Responses

-   **`200 OK`**: Analysis task successfully initiated.

    ```json
    {
      "job_id": "string-uuid-of-the-job"
    }
    ```

## 3. Endpoint: `GET /status/{job_id}`

Retrieves the current status and progress of a video analysis job. This endpoint is designed for polling by the frontend to update the user interface.

- **URL**: `/status/{job_id}`
- **Method**: `GET`
- **Description**: Fetches the status, progress, and results (if completed) of a specific analysis job.

### Path Parameters

| Name     | Type   | Description                                | Example               |
| :------- | :----- | :----------------------------------------- | :-------------------- |
| `job_id` | `str`  | The unique identifier returned by `/upload`. | `a1b2c3d4-e5f6-7890-1234-567890abcdef` |

### Polling Mechanism

Clients should periodically call this endpoint (e.g., every 1-5 seconds) using the `job_id` to get updates on the analysis progress.

### Responses

-   **`200 OK`**: Returns the job status, progress, and results if available.

    **Processing State**
    ```json
    {
      "status": "processing",
      "progress": 45  // Percentage (0-100)
    }
    ```

    **Completed State**
    ```json
    {
      "status": "completed",
      "progress": 100,
      "result": {
        "video_url": "http://localhost:8000/outputs/analyzed_my_match.mp4",
        "threat_data": [
          { "time": 0.0, "value": 0.0 },
          { "time": 1.5, "value": 10.2 },
          // ... more threat data ...
        ],
        "passes": [
          {
            "frame": 450,
            "passer_id": 12,
            "target_id": 7,
            "type": "through",
            "score": 0.88,
            "meta": { ... }
          }
          // ... more pass predictions ...
        ]
      }
    }
    ```

    **Failed State**
    ```json
    {
      "status": "failed",
      "progress": 0 // Or last known progress
    }
    ```

-   **`404 Not Found`**: If the provided `job_id` does not exist.

## 4. Static File Serving: `/outputs`

The backend serves all files located in the `output_videos` directory (relative to the backend's root) under the `/outputs` URL path.

- **Base URL**: `http://localhost:8000/outputs/`
- **Description**: Access analyzed videos, generated heatmaps, and other output artifacts directly via HTTP.
- **Examples**:
    - `http://localhost:8000/outputs/analyzed_my_match.mp4`
    - `http://localhost:8000/outputs/heatmaps/player_123.png`
