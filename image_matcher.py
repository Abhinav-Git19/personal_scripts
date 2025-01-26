import cv2
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_all_image_paths(folder_path, prefix_filter=None):
    """
    Recursively collects all image file paths from the given folder.
    Optionally filters files based on a prefix.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                if prefix_filter is None or file.startswith(prefix_filter):
                    image_paths.append(os.path.join(root, file))
    return image_paths


def resize_image(image, target_dim=500):
    """
    Resizes the image so its smaller dimension equals target_dim while maintaining the aspect ratio.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    if h > w:
        new_h = target_dim
        new_w = int((w / h) * target_dim)
    else:
        new_w = target_dim
        new_h = int((h / w) * target_dim)
    return cv2.resize(image, (new_w, new_h))


def match_images(query_image_path, folder_images, threshold=30, target_dim=500, chunk_size=100):
    """
    Matches a single query image with Folder 1 images in chunks and returns the best match.
    """
    try:

        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params = {
            "algorithm": FLANN_INDEX_LSH,
            "table_number": 12,  # Number of hash tables
            "key_size": 20,      # Size of the hash keys
            "multi_probe_level": 2,  # Number of probes
        }

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Read and preprocess the query image
        query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_image is None:
            return query_image_path, None
        query_image = resize_image(query_image, target_dim)

        query_keypoints, query_descriptors = orb.detectAndCompute(query_image, None)
        if query_descriptors is None:
            return query_image_path, None

        best_match = None
        best_distance = float('inf')

        # Process Folder 1 images in chunks
        for i in range(0, len(folder_images), chunk_size):
            folder_image_chunk = folder_images[i:i + chunk_size]
            for folder_image_path in folder_image_chunk:
                folder_image = cv2.imread(folder_image_path, cv2.IMREAD_GRAYSCALE)
                if folder_image is None:
                    continue
                folder_image = resize_image(folder_image, target_dim)

                folder_keypoints, folder_descriptors = orb.detectAndCompute(folder_image, None)
                if folder_descriptors is None:
                    continue

                matches = bf.match(query_descriptors, folder_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                if matches and matches[0].distance < threshold and matches[0].distance < best_distance:
                    best_match = folder_image_path
                    best_distance = matches[0].distance

        return query_image_path, best_match
    except Exception as e:
        return query_image_path, None


def append_results_to_file(results, output_file):
    """
    Appends new results to the JSON file.
    """
    try:
        # Load existing results if the file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                existing_results = json.load(file)
        else:
            existing_results = {}

        # Update existing results with new results
        existing_results.update(results)

        # Save back to file
        with open(output_file, "w") as file:
            json.dump(existing_results, file, indent=4)
    except Exception as e:
        print(f"Error updating results to file: {e}")


def match_images_multiple_runs(folder1, folder2, num_runs=2, threshold=30, target_dim=500, chunk_size=100, max_workers=4, output_file="matches.json"):
    """
    Matches images from Folder 2 with images in Folder 1 using parallelization.
    Refines the search space for each query image in subsequent runs based on previous matches.
    Writes partial results to a JSON file after each run.
    """
    folder1_images = get_all_image_paths(folder1)
    folder2_images = get_all_image_paths(folder2, prefix_filter="PHOTO")  # Focus only on files starting with PHOTO

    match_results = {query_image: folder1_images for query_image in folder2_images}

    print("Starting refined image matching...")
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        refined_results = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(match_images, query_image, match_results[query_image], threshold, target_dim, chunk_size): query_image
                for query_image in folder2_images
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Run {run + 1}"):
                query_image_path, best_match = future.result()
                refined_results[query_image_path] = best_match

                # Save partial results to file
                append_results_to_file({query_image_path: best_match}, output_file)

        # Refine search space for the next run
        match_results = {query: [match] for query, match in refined_results.items() if match}

    print(f"Final matching results saved to {output_file}")


if __name__ == '__main__':
    # Define folders
    folder1 = "reference_photos"
    folder2 = "album_groom"
    output_file = "image_matches.json"

    # Run matching
    match_images_multiple_runs(
        folder1, folder2,
        num_runs=2,         # Limit to 2 runs for faster processing
        threshold=30,       # Distance threshold for matches
        target_dim=500,     # Resize dimension for images
        chunk_size=300,     # Size of chunks for Folder 1
        max_workers=8,      # Number of workers for parallelization
        output_file=output_file
    )

    print(f"Matching process completed. Results saved in {output_file}")
