

<h2 align="center"><strong>S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing</strong></h2>

<div align="center">
  <h5>
    <em>Liang Lv<sup>1</sup>, Di Wang<sup>1</sup>, Jing Zhang<sup>1 †</sup>, Lefei Zhang<sup>1 †</sup>, Bo Du<sup>1</sup>, Liangpei Zhang<sup>1</sup></em>
    <br><br>
    <sup>1</sup> Wuhan University, China
  </h5>
</div>


<hr>


# 📚 Contents

- [Abstract](#abstract)

<hr>


# 📄 Abstract

Semi-supervised semantic segmentation (S4) has advanced remote sensing (RS) analysis by leveraging unlabeled data through pseudo-labeling and consistency learning. However, existing RS studies often rely on small-scale datasets and models, limiting their practical applicability. To address this, we propose S5, the first scalable framework for semi-supervised semantic segmentation in RS, which unlocks the potential of vast unlabeled Earth observation data typically underutilized due to costly pixel-level annotations. Our approach introduces MillionSeg, a novel dataset comprising over one million unlabeled RS images spanning diverse geospatial scenes, and systematically scales S4 methods by pre-training RS foundation models of varying sizes on this extensive corpus, generating high-quality pseudo-labels as a byproduct. Through experiments on extensive benchmarks of diverse RS tasks involving semantic segmentation, object detection, and change detection, we show that S5 effectively learns generalizable representations, and scales model capacity to enhance the performance of various downstream RS tasks. The resulting foundation models achieve state-of-the-art performance across all benchmarks, underscoring the viability of scaling semi-supervised learning for RS applications.

<hr>