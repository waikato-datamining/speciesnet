import io
import PIL.ImageOps

from datetime import datetime
from typing import Dict, Tuple
from PIL import Image
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from speciesnet import DEFAULT_MODEL, SUPPORTED_MODELS, SpeciesNet
from speciesnet.utils import load_rgb_image
from speciesnet.utils import BBox as SNBBox


def load_model(model_name: str = DEFAULT_MODEL) -> SpeciesNet:
    """
    Loads the specified speciesnet model.

    :param model_name: the name of the model
    :type model_name: str
    :return: the SpeciesNet instance
    :rtype: SpeciesNet
    """
    if model_name not in SUPPORTED_MODELS:
        raise Exception("Unsupported model (%s): %s" % ("|".join(SUPPORTED_MODELS), model_name))
    return SpeciesNet(model_name, geofence=False, multiprocessing=False)


def predict(img, model: SpeciesNet) -> Tuple[Dict, int, int]:
    """
    Generates a prediction for the image.

    :param img: the image (file path or bytes)
    :param model: the SpeciesNet instance to use
    :type model: SpeciesNet
    :return: the generate prediction
    :rtype: tuple of predictions, width and height
    """
    if isinstance(img, str):
        image = load_rgb_image(img)
    elif isinstance(img, bytes):
        image = Image.open(io.BytesIO(img))
        image = image.convert("RGB")
        image = PIL.ImageOps.exif_transpose(image)
    else:
        raise Exception("Either file path or bytes expected, received: %s" % str(type(img)))

    filepath = str(datetime.now())
    width, height = image.size
    detector_results = dict()
    classifier_results = dict()

    # run detector
    detector_input = model.detector.preprocess(image)
    detector_results[filepath] = model.detector.predict(filepath, detector_input)

    # preprocess image for classifier
    detections = detector_results[filepath].get("detections", None)
    if detections:
        bboxes = [SNBBox(*det["bbox"]) for det in detections]
    else:
        bboxes = []
    classifier_input = model.classifier.preprocess(image, bboxes=bboxes)

    # run classifier
    classifier_results[filepath] = model.classifier.predict(filepath, classifier_input)

    # combine results
    ensemble_results = model.ensemble.combine(
        filepaths=[filepath],
        classifier_results=classifier_results,
        detector_results=detector_results,
        geolocation_results=dict(),
        partial_predictions=dict(),
    )
    return {"predictions": ensemble_results}, width, height


def prediction_to_file(predictions, id_: str, path: str, threshold: float = 0.5, width: int = None, height: int = None) -> str:
    """
    Saves the predictions as OPEX in the specified file. 

    :param predictions: the predictions to save
    :param id_: the ID for the OPEX output
    :type id_: str
    :param path: the file to save the predictions to
    :type path: str
    :param threshold: the minimum score for retaining predictions
    :type threshold: float
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :return: the file the predictions were saved to
    :rtype: str
    """
    data = prediction_to_data(predictions, id_, threshold=threshold, width=width, height=height)
    with open(path, "w") as fp:
        fp.write(data)
        fp.write("\n")
    return path


def prediction_to_data(predictions, id_: str, threshold: float = 0.5, width: int = None, height: int = None) -> str:
    """
    Turns the predictions into an OPEX string.

    :param predictions: the predictions to convert
    :param id_: the ID for the OPEX output
    :type id_: str
    :param threshold: the minimum score for retaining predictions
    :type threshold: float
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :return: the generated predictions
    :rtype: str
    """
    pred_objs = []
    if "predictions" in predictions:
        for prediction in predictions["predictions"]:
            if ("detections" not in prediction) or (len(prediction["detections"]) == 0):
                continue
            score = None
            if "prediction_score" in prediction:
                score = prediction["prediction_score"]
            if (score is not None) and (threshold is not None):
                if score < threshold:
                    continue
            meta = dict()
            meta["model_version"] = prediction["model_version"]
            label = None
            if "prediction" in prediction:
                label = prediction["prediction"]
            xmin_n, ymin_n, width_n, height_n = prediction["detections"][0]["bbox"]
            xmin = int(xmin_n * width)
            ymin = int(ymin_n * height)
            xmax = xmin + int(width_n * width) - 1
            ymax = ymin + int(height_n * height) - 1
            bbox = BBox(left=xmin, top=ymin, right=xmax, bottom=ymax)
            poly = Polygon(points=[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            opex_obj = ObjectPrediction(label=label, score=score, bbox=bbox, polygon=poly)
            pred_objs.append(opex_obj)
    preds = ObjectPredictions(id=id_, timestamp=str(datetime.now()), objects=pred_objs)
    return preds.to_json_string()
