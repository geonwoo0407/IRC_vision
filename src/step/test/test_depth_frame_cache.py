"""Tests for the unified aligned-depth frame cache."""

import numpy as np

from step.depth_frame_cache import DepthFrameCache
from step.depth_frame_cache import DepthFrameConsumer


class _Consumer(DepthFrameConsumer):
    pass


def test_shared_consumers_read_the_same_depth_frame():
    """One cache update is immediately visible to every analyzer consumer."""
    cache = DepthFrameCache()
    first = _Consumer()
    second = _Consumer()
    first.depth_cache = cache
    second.depth_cache = cache
    image = np.full((2, 3), 900, dtype=np.uint16)

    cache.update(image, width=3, height=2, received_at=12.5)

    assert first.latest_depth_image is image
    assert second.latest_depth_image is image
    assert first.latest_depth_time == 12.5
    assert second.latest_image_width == 3
    assert second.latest_image_height == 2


def test_consumer_properties_still_support_lightweight_unit_helpers():
    """Analyzer tests may populate depth fields without constructing ROS."""
    consumer = _Consumer()
    image = np.ones((1, 1), dtype=np.uint16)

    consumer.latest_depth_image = image
    consumer.latest_depth_time = 3.0

    assert consumer.latest_depth_image is image
    assert consumer.latest_depth_time == 3.0
