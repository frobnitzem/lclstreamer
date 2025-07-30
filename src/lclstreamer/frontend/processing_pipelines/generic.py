from collections.abc import Iterator
import numpy as np

from ...models.parameters import Parameters, NoOpPipelineParameters, BatchPipelineParameters
from ...protocols.backend import StrFloatIntNDArray
from ...protocols.frontend import ProcessingPipelineProtocol
from .utils import DataStorage


class NoOpPipeline(ProcessingPipelineProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(self, parameters: NoOpPipelineParameters) -> None:
        """
        Initializes a NoOp processing pipeline

        This pipeline performs no operations on the data.
        It simply forwards each item.

        Arguments:

             parameters: The configuration parameters
        """
        pass

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray]]
    ) -> Iterator[dict[str, StrFloatIntNDArray]]:
        """
        Processes a single data event and forward the results.

        Since this is a NoOp processing pipeline, this function
        simply forwards data from each event without performing
        any processing

        Arguments:

            data: A dictionary storing data belonging to a data event.

        Returns:

            data: The same data provided to the function as an input
        """

        # FIXME: currently we need to add 1 dimension at start so the serializer works
        #        but this should not be needed (need better serializer)
        for data in stream:
            yield {key:np.array([val]) for key,val in data.items()}

class BatchPipeline(ProcessingPipelineProtocol):
    """
    Initializes a Batching pipeline
    This pipeline collects data into batches.

     Arguments:

         parameters: The configuration parameters
    """
    def __init__(self, parameters: BatchPipelineParameters) -> None:
        self.batch_size = parameters.batch_size

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray]]
    ) -> Iterator[dict[str, StrFloatIntNDArray]]:
        """
        Process a stream of input data and batch
        together their results.

        Arguments:
             data: A dictionary storing data belonging to a data event.

         Returns:
            data: A dictionary storing grouped event data belonging
                  to data events. First output dimension is the
                  number of accumulated events.
         """
        data_storage = DataStorage()

        for data in stream:
            data_storage.add_data(data=data)

            if len(data_storage) >= self.batch_size:
                yield data_storage.retrieve_stored_data()
                data_storage.reset_data_storage()

        if len(data_storage) > 0:
            yield data_storage.retrieve_stored_data()
