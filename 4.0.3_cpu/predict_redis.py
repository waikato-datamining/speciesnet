from datetime import datetime
import traceback

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, predict, prediction_to_data
from speciesnet import DEFAULT_MODEL, SUPPORTED_MODELS


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        data = msg_cont.message['data']
        preds, width, height = predict(data, config.model)
        out_data = prediction_to_data(preds, str(start_time), threshold=config.threshold, width=width, height=height)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('PaddleDetection - Prediction (Redis)', prog="paddledet_predict_redis", prefix="redis_")
    parser.add_argument('--model', help='The model to use', choices=SUPPORTED_MODELS, required=False, default=DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, help='The score threshold for predictions', required=False, default=0.5)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model = load_model(parsed.model)

        config = Container()
        config.model = model
        config.threshold = parsed.threshold
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())

