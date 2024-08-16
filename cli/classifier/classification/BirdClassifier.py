"""Classify birds from images."""


import logging
import configparser
import multiprocessing
import tensorflow

from aux.timec import timec
from classification.classification import Classification, ClassificationFatalException


class _StopAllException(Exception):
    """Exception signifying that everything should be stopped for a given process."""


class _StopTaskException(Exception):
    """Exception signifying that the given task should be stopped."""


class _BirdClassifier:
    """Handle BirdClassifierTasks."""

    def __init__(self, args):
        self.args = args

    def load(self, args):
        bird_model = None
        try:
            with timec() as t:
                bird_model = Classification.load_model(self.args['url_model'])
            if self.args['time']: logging.debug(f"Time taken for model load: {t():.4f}s")
        except ClassificationFatalException:
            logging.error("Failed to load model - stopping")
            raise _StopAllException
        bird_labels = None
        try:
            with timec() as t:
                bird_labels = Classification.load_labels(self.args['url_labels'])
            if self.args['time']: logging.debug(f"Time taken for labels load: {t():.4f}s")
        except ClassificationFatalException:
            logging.error("Failed to load labels - stopping")
            raise _StopAllException
        return bird_model, bird_labels

    def handle_task(self, task, bird_model, bird_labels):
        try:
            image_array = Classification.load_image(task.image)
            image = Classification.format_image(image_array)
            image_tensor = Classification.generate_tensor(image)
            try:
                with timec() as t:
                    # call() calls the model on new inputs:
                    # "In this case call just reapplies all ops in the graph to the new inputs
                    # (e.g. build a new computational graph from the provided inputs)."
                    model_raw_output = bird_model.call(image_tensor).numpy()
                if self.args['time']: logging.debug(f"Time taken for model call: {t():.4f}s")
            except tensorflow.python.framework.errors_impl.InvalidArgumentError:
                raise _StopTaskException
            birds_names_with_results_ordered = Classification.order_by_result_score(
                    model_raw_output, bird_labels)

            return [ Classification.get_top_n_result(i, birds_names_with_results_ordered)
                    for i in range(1, self.args['nm_top_results'] + 1) ]
        except ClassificationFatalException:
            raise _StopTaskException


class _BirdClassifierMain(_BirdClassifier):
    """Handle BirdClassifierTasks in the main process."""

    def __init__(self, tasks, args):
        _BirdClassifier.__init__(self, args)
        self.tasks = tasks

    def run(self):
        try:
            bird_model, bird_labels = self.load(self.args)
        except _StopAllException:
            return [ _BirdClassifierResponse(x, "fixme", None) for x in range(len(self.tasks)) ]

        answers = []
        for task in self.tasks:
            answer = None
            try:
                answer = _BirdClassifierResponse(task.index, task.image,
                        self.handle_task(task, bird_model, bird_labels))
            except _StopTaskException:
                logging.debug("Stopping task %s", str(task))
                answer = _BirdClassifierResponse(task.index, task.image, None)
            except Exception:
                logging.exception("Unexpected error when handling task %s", str(task))
                answer = _BirdClassifierResponse(task.index, task.image, None)
            answers += [answer]

        return answers


class _BirdClassifierWorker(multiprocessing.Process, _BirdClassifier):
    """Worker process to handle BirdClassifierTasks in parallel."""

    def __init__(self, name, task_queue, result_queue, args):
        multiprocessing.Process.__init__(self)
        _BirdClassifier.__init__(self, args)
        self.name = name
        self.task_queue = task_queue
        self.result_queue = result_queue

    def _debug_log(self, msg, *args, **kwargs):
        logging.debug("%s: %s", multiprocessing.current_process().name, msg, *args, **kwargs)

    def _answer(self, answer):
        self.task_queue.task_done()
        self.result_queue.put(answer)

    def run(self):
        multiprocessing.current_process().name = self.name
        self._debug_log("Running")

        # Note: This should be shared memory resources between the processes.
        # multiprocessing.shared_memory, introduced in Python3.8, could potentially be used here.
        try:
            bird_model, bird_labels = self.load(self.args)
        except _StopAllException:
            return

        while True:
            task = None
            response = None
            try:
                task = self.task_queue.get()
                if task is None:
                    self._debug_log("Exiting")
                    self.task_queue.task_done()
                    break
                self._debug_log("Got task %s", str(task))
                response = _BirdClassifierResponse(task.index, task.image,
                        self.handle_task(task, bird_model, bird_labels))
                self._answer(response)
            except _StopTaskException:
                self._debug_log("Stopping task %s", str(task))
                response = _BirdClassifierResponse(task.index, task.image, None)
                self._answer(response)
            except Exception:
                logging.exception("Unexpected error when handling task %s", str(task))
                response = _BirdClassifierResponse(task.index, task.image, None)
                self._answer(response)


class _BirdClassifierTask:
    """Bird classification task definition."""

    def __init__(self, index, image):
        self.index = index
        self.image = image

    def __str__(self):
        return "({}, {})".format(self.index, self.image)


class _BirdClassifierResponse:
    """
    Bird classification answer.

    This has the format (index, [top n results]) where the latter will be None when a
    classification could not be fetched.
    """

    def __init__(self, index, image, classifications):
        self.index = index
        self.image = image
        self.classifications = classifications

    def __str__(self):
        classifications = None
        if not self.classifications:
            classifications = [str(self.classifications)]
        else:
            classifications = [ str(x) for x in self.classifications ]
        return "{}\n    {}".format(self.image, "\n    ".join(classifications))

    def __lt__(self, other):
        return self.index < other.index


def _classify_birds_main(image_urls, args):
    tasks = [ _BirdClassifierTask(i, image) for i, image in enumerate(image_urls) ]
    return _BirdClassifierMain(tasks, args).run()


def _classify_birds_multiprocessing(image_urls, nm_tasks, args):
    tasks = multiprocessing.JoinableQueue()
    responses = multiprocessing.Queue()

    for i, image in enumerate(image_urls):
        tasks.put(_BirdClassifierTask(i, image))
    logging.debug("All tasks put to queue")

    nm_workers = min(nm_tasks, multiprocessing.cpu_count())
    for i in range(nm_workers):
        tasks.put(None)
    logging.debug("Creating {} worker(s)".format(nm_workers))
    workers = [
            _BirdClassifierWorker("BirdClassifierWorker-{}".format(i), tasks, responses, args)
            for i in range(nm_workers) ]
    for worker in workers:
        worker.start()
    logging.debug("All worker(s) started")

    tasks.join()

    results = []
    while nm_tasks:
        resp = responses.get()
        logging.debug("Got response: %s", str(resp))
        results += [resp]
        nm_tasks -= 1

    return sorted(results)


def classify_birds(image_urls, config, args):
    """
    Classify birds from a list of image URLs.

    Parameters:
        image_urls ([str]): A list of image URLs
        config (ConfigParser): A ConfigParser containing application configuration values

    Returns:
        answers ([BirdClassifierResponse]): A list of answers in the order the image URLs
                                            arrived
    """

    nm_tasks = len(image_urls)

    if nm_tasks < args['multiprocessing_threshold']:
        logging.debug("nm_tasks=%d < multiprocessing_threshold=%d - running in main process",
                nm_tasks, args['multiprocessing_threshold'])
        return _classify_birds_main(image_urls, args)
    else:
        logging.debug("nm_tasks=%d >= multiprocessing_threshold=%d - starting worker processes",
                nm_tasks, args['multiprocessing_threshold'])
        return _classify_birds_multiprocessing(image_urls, nm_tasks, args)
