<?php
$result = null;
$error = null;

// Ensure the uploads directory exists
$upload_dir = 'uploads/';
if (!is_dir($upload_dir)) {
    mkdir($upload_dir, 0777, true);
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['xray_image'])) {
    $file_name = basename($_FILES["xray_image"]["name"]);
    // Use timestamps to prevent overwriting files with the same name
    $target_file = $upload_dir . time() . '_' . $file_name;
    $heatmap_file = $upload_dir . 'heatmap_' . time() . '_' . $file_name;

    if (move_uploaded_file($_FILES["xray_image"]["tmp_name"], $target_file)) {
        // Build the command to run the Python script
        $command = escapeshellcmd("python ../scripts/inference_script_gui.py " . escapeshellarg($target_file) . " " . escapeshellarg($heatmap_file));
        
        // Execute the command and capture the JSON output
        $output = shell_exec($command);
        
        if ($output) {
            $result = json_decode($output, true);
            $result['original'] = $target_file;
            
            if (isset($result['error'])) {
                $error = $result['error'];
            }
        } else {
            $error = "Failed to execute the Python script. Check your environment.";
        }
    } else {
        $error = "Failed to upload the image.";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical AI: Pulmonary Disease Detection</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 2rem; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .upload-section { text-align: center; margin-bottom: 2rem; padding: 2rem; border: 2px dashed #cbd5e1; border-radius: 8px; }
        input[type="file"] { margin-bottom: 1rem; }
        button { background-color: #3b82f6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 1rem; }
        button:hover { background-color: #2563eb; }
        .results-section { margin-top: 2rem; }
        .diagnosis-card { padding: 1.5rem; border-left: 5px solid; border-radius: 4px; margin-bottom: 2rem; color: #1e293b; }
        .status-normal { background-color: #dcfce7; border-color: #22c55e; }
        .status-pneumonia { background-color: #fee2e2; border-color: #ef4444; }
        .status-attention { background-color: #fef08a; border-color: #eab308; }
        .images-container { display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; }
        .image-box { flex: 1; min-width: 300px; text-align: center; }
        .image-box img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .error { color: #ef4444; background: #fee2e2; padding: 1rem; border-radius: 4px; }
    </style>
</head>
<body>

<div class="container">
    <h1>Pulmonary Disease Detection AI</h1>
    
    <div class="upload-section">
        <form action="index.php" method="POST" enctype="multipart/form-data">
            <h3>Upload a Chest X-Ray</h3>
            <input type="file" name="xray_image" accept="image/jpeg, image/png" required>
            <br>
            <button type="submit">Analyze Image</button>
        </form>
    </div>

    <?php if ($error): ?>
        <div class="error"><?php echo htmlspecialchars($error); ?></div>
    <?php endif; ?>

    <?php if ($result && !isset($result['error'])): ?>
        <?php 
            // Determine the correct CSS class based on the diagnosis
            $status_class = 'status-attention'; // Default to yellow
            if ($result['diagnosis'] === 'NORMAL') {
                $status_class = 'status-normal';
            } elseif ($result['diagnosis'] === 'PNEUMONIA') {
                $status_class = 'status-pneumonia';
            }
        ?>
        
        <div class="results-section">
            <!-- Inject the dynamic CSS class here -->
            <div class="diagnosis-card <?php echo $status_class; ?>">
                <h2>Diagnosis: <strong><?php echo $result['diagnosis']; ?></strong></h2>
                <p><strong>Confidence:</strong> <?php echo $result['confidence']; ?>%</p>
                <p><small>Normal Probability: <?php echo $result['normal_prob']; ?>% | Pneumonia Probability: <?php echo $result['pneumonia_prob']; ?>%</small></p>
            </div>

            <div class="images-container">
                <div class="image-box">
                    <h3>Original X-Ray</h3>
                    <img src="<?php echo $result['original']; ?>" alt="Original X-Ray">
                </div>
                <div class="image-box">
                    <h3>Grad-CAM Analysis (Culprit Area)</h3>
                    <img src="<?php echo $result['heatmap_path']; ?>" alt="AI Heatmap">
                </div>
            </div>
        </div>
    <?php endif; ?>
</div>

</body>
</html>