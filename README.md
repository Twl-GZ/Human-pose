# Local Feature Enhancement for Robust 2D Multi-Person Pose Estimation via Posture Refinement Network

## Introduction
This paper proposes a novel posture refinement network that leverages local feature enhancement and fusion to address these limitations. The network employs HRNet as the backbone to extract multi-scale feature maps, introducing a Dilated Convolution Module (DCM) with cascaded dilated convolutions to enrich pose keypoint representations. Additionally, a Hybrid Self-Attention Module (HSM) integrates contextual information, further refining pose estimates.

The rest of the code is being refined and updated

		
## Main Results
### Results on COCO val2017 without multi-scale test
| Backbone | Input size | AP | AP .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 0.689 | 0.874 | 0.752 | 0.629 | 0.779 |
| **pose_hrnet_w48** |  640x640 | 0.715 | 0.887 | 0.778 | 0.672 | 0.791 |

### Results on COCO val2017 with multi-scale test
| Backbone | Input size | AP | AP .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 0.713 | 0.882 | 0.775 | 0.669 | 0.785 |
| **pose_hrnet_w48** |  640x640 | 0.731 | 0.889 | 0.791 | 0.695 | 0.790 |

### Results on COCO test-dev2017 without multi-scale test
| Backbone | Input size | AP | AP .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 0.679 | 0.884 | 0.745 | 0.618 | 0.766 |
| **pose_hrnet_w48** |  640x640 | 0.708 | 0.896 | 0.778 | 0.663 | 0.776 |

### Results on COCO test-dev2017 with multi-scale test
| Backbone | Input size | AP | AP .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 0.702 | 0.893 | 0.772 | 0.655 | 0.768 |
| **pose_hrnet_w48** |  640x640 | 0.718 | 0.897 | 0.790 | 0.678 | 0.777 |

### Results on CrowdPose test without multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.673 | 0.870 | 0.724 | 0.746 | 0.681 | 0.590 |
| **pose_hrnet_w48** | 0.687 | 0.873 | 0.737 | 0.750 | 0.691 | 0.604 |

### Results on CrowdPose test with multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.685 | 0.859 | 0.738 | 0.763 | 0.693 | 0.584 |
| **pose_hrnet_w48** | 0.691 | 0.860 | 0.746 | 0.771 | 0.698 | 0.592 |






