# graphld.clumping

LD clumping for identifying independent variants.

LD clumping identifies independent index variants by iteratively selecting the variant with the highest χ² statistic and pruning all variants in high LD with it. Clumping + thresholding is a popular (though suboptimal) way of computing polygenic scores.

For usage examples, see the [LD Clumping guide](../python_api/clumping.md).

::: graphld.clumping
    options:
      show_root_heading: true
      members_order: source
