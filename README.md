# Local Feature Enhancement for Robust 2D Multi-Person Pose Estimation via Poseture Refinement Networks

## Introduction
This paper proposes a novel poseture refinement network that leverages local feature enhancement and fusion to address these limitations. The network employs HRNet as the backbone to extract multi-scale feature maps, introducing a Dilated Convolution Module (DCM) with cascaded dilated convolutions to enrich pose keypoint representations. Additionally, a Hybrid Self-Attention Module (HSM) integrates contextual information, further refining pose estimates.

		
## Main Results
### Results on COCO val2017 without multi-scale test
| Backbone | Input size | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 45.4 | 0.707 | 0.877 | 0.771 | 0.662 | 0.778 | 0.759 | 0.913 | 0.813 | 0.705 | 0.836 |
| **pose_hrnet_w48** |  640x640 | 141.5 | 0.723 | 0.883 | 0.786 | 0.686 | 0.786 | 0.777 | 0.924 | 0.832 | 0.728 | 0.849 |

### Results on COCO val2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.707 | 0.877 | 0.771 | 0.662 | 0.778 | 0.759 | 0.913 | 0.813 | 0.705 | 0.836 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.723 | 0.883 | 0.786 | 0.686 | 0.786 | 0.777 | 0.924 | 0.832 | 0.728 | 0.849 |

### Results on COCO test-dev2017 without multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.673 | 0.879 | 0.741 | 0.615 | 0.761 | 0.724 | 0.908 | 0.782 | 0.654 | 0.819 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.700 | 0.894 | 0.773 | 0.657 | 0.769 | 0.754 | 0.927 | 0.816 | 0.697 | 0.832 |

### Results on COCO test-dev2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.698 | 0.890 | 0.766 | 0.652 | 0.765 | 0.751 | 0.924 | 0.811 | 0.695 | 0.828 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.710 | 0.892 | 0.780 | 0.671 | 0.769 | 0.767 | 0.932 | 0.830 | 0.715 | 0.839 |

### Results on CrowdPose test without multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.657 | 0.857 | 0.704 | 0.730 | 0.664 | 0.575 |
| **pose_hrnet_w48** | 0.673 | 0.864 | 0.722 | 0.746 | 0.681 | 0.587 |

### Results on CrowdPose test with multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.670 | 0.854 | 0.724 | 0.755 | 0.680 | 0.569 |
| **pose_hrnet_w48** | 0.680 | 0.855 | 0.734 | 0.766 | 0.688 | 0.584 |






