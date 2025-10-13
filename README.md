# ACE-G: Improving Scene Coordinate Regression Through Query Pre-Training

<div align="center">

[Leonard Bruns](https://roym899.github.io/), [Axel Barroso-Laguna](https://scholar.google.com/citations?user=m_SPRGUAAAAJ), [Tommaso Cavallari](https://scholar.google.it/citations?user=r7osSm0AAAAJ), [Áron Monszpart](https://amonszpart.github.io/), [Sowmya Munukutla](https://scholar.google.com/citations?user=l-zRzDEAAAAJ), [Victor Adrian Prisacariu](https://www.robots.ox.ac.uk/~victor/), [Eric Brachmann](https://ebrach.github.io/)

![ACE-G](./resources/teaser.gif)

</div>

> Scene coordinate regression (SCR) has established itself as a promising learning-based approach to visual relocalization. After mere minutes of scene-specific training, SCR models estimate camera poses of query images with high accuracy. Still, SCR methods fall short of the generalization capabilities of more classical feature-matching approaches. When imaging conditions of query images, such as lighting or viewpoint, are too different from the training views, SCR models fail. Failing to generalize is an inherent limitation of previous SCR frameworks, since their training objective is to encode the training views in the weights of the coordinate regressor itself. The regressor essentially overfits to the training views, by design. We propose to separate the coordinate regressor and the map representation into a generic transformer and a scene-specific map code. This separation allows us to pre-train the transformer on tens of thousands of scenes. More importantly, it allows us to train the transformer to generalize from mapping images to unseen query images during pre-training. We demonstrate on multiple challenging relocalization datasets that our method, ACE-G, leads to significantly increased robustness while keeping the computational footprint attractive.

Code coming soon! In the meantime, check out our [project page](https://ace-g.github.io/).