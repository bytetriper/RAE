# RAE OT figure assets

This package collects the paper-ready plots, grids, spatial maps, and LaTeX rows for the decoder/interface autopsy.

- Pairing note: index-aligned by filename order

## LaTeX snippets

### paired_vs_shuffle_mse

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/paired_vs_shuffle_mse.pdf}
  \caption{Paired latent recovery is substantially stronger than class-shuffled or globally shuffled controls, indicating that generated images preserve instance-level latent identity after re-encoding.}
  \label{fig:paired_vs_shuffle_mse}
\end{figure}
```

### paired_mse_ratios

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/paired_mse_ratios.pdf}
  \caption{Shuffled-to-paired recovery ratios emphasize how much closer the generated images are to their paired latents than to trivial controls.}
  \label{fig:paired_mse_ratios}
\end{figure}
```

### paired_error_hist_pooled

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/paired_error_hist_pooled.pdf}
  \caption{Per-sample paired error distributions show that the recovery improvement is broad rather than driven by a handful of outliers.}
  \label{fig:paired_error_hist_pooled}
\end{figure}
```

### paired_error_hist_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/paired_error_hist_spatial_jl.pdf}
  \caption{Per-sample paired error distributions show that the recovery improvement is broad rather than driven by a handful of outliers.}
  \label{fig:paired_error_hist_spatial_jl}
\end{figure}
```

### pca_scatter_pooled

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/pca_scatter_pooled.pdf}
  \caption{The saved and re-encoded latent clouds occupy overlapping regions in the first two principal components, suggesting approximate recovery with some residual shift.}
  \label{fig:pca_scatter_pooled}
\end{figure}
```

### pca_scatter_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/pca_scatter_spatial_jl.pdf}
  \caption{The saved and re-encoded latent clouds occupy overlapping regions in the first two principal components, suggesting approximate recovery with some residual shift.}
  \label{fig:pca_scatter_spatial_jl}
\end{figure}
```

### pca_arrows_pooled

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/pca_arrows_pooled.pdf}
  \caption{Paired arrows show that most recovered latents move a short distance, with a mix of local drift and compression rather than arbitrary global relocation.}
  \label{fig:pca_arrows_pooled}
\end{figure}
```

### pca_arrows_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/pca_arrows_spatial_jl.pdf}
  \caption{Paired arrows show that most recovered latents move a short distance, with a mix of local drift and compression rather than arbitrary global relocation.}
  \label{fig:pca_arrows_spatial_jl}
\end{figure}
```

### bootstrap_metric_calibration

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/bootstrap_metric_calibration.pdf}
  \caption{Sinkhorn stays near the self-split floor, while sliced W2 remains clearly above it, which is the main distribution-level mismatch signal.}
  \label{fig:bootstrap_metric_calibration}
\end{figure}
```

### worst_sliced_projection_histograms

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/worst_sliced_projection_histograms.pdf}
  \caption{The top sliced-W2 directions reveal whether mismatch is driven by mean shift, variance shrinkage, or tail distortion along a few directions.}
  \label{fig:worst_sliced_projection_histograms}
\end{figure}
```

### worst_sliced_projection_histograms_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/worst_sliced_projection_histograms_spatial_jl.pdf}
  \caption{The same worst-projection view for the spatial-JL embedding checks whether the mismatch is a pooled-only artifact.}
  \label{fig:worst_sliced_projection_histograms_spatial_jl}
\end{figure}
```

### embedding_norm_hist_pooled

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/embedding_norm_hist_pooled.pdf}
  \caption{Norm distributions diagnose contraction or expansion of the recovered latent cloud after re-encoding.}
  \label{fig:embedding_norm_hist_pooled}
\end{figure}
```

### embedding_norm_hist_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/embedding_norm_hist_spatial_jl.pdf}
  \caption{Norm distributions diagnose contraction or expansion of the recovered latent cloud after re-encoding.}
  \label{fig:embedding_norm_hist_spatial_jl}
\end{figure}
```

### per_dimension_variance_pooled

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/per_dimension_variance_pooled.pdf}
  \caption{Per-dimension variance comparisons show whether re-encoding collapses or inflates particular latent directions.}
  \label{fig:per_dimension_variance_pooled}
\end{figure}
```

### per_dimension_variance_spatial_jl

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/per_dimension_variance_spatial_jl.pdf}
  \caption{Per-dimension variance comparisons show whether re-encoding collapses or inflates particular latent directions.}
  \label{fig:per_dimension_variance_spatial_jl}
\end{figure}
```

### best_recovered_examples_grid

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{image_grids/pdf/best_recovered_examples_grid.pdf}
  \caption{Selected generated-image examples illustrate how latent recovery quality varies across easy, hard, semantically good, and semantically bad cases.}
  \label{fig:best_recovered_examples_grid}
\end{figure}
```

### worst_recovered_examples_grid

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{image_grids/pdf/worst_recovered_examples_grid.pdf}
  \caption{Selected generated-image examples illustrate how latent recovery quality varies across easy, hard, semantically good, and semantically bad cases.}
  \label{fig:worst_recovered_examples_grid}
\end{figure}
```

### semantic_good_spatial_bad_grid

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{image_grids/pdf/semantic_good_spatial_bad_grid.pdf}
  \caption{Selected generated-image examples illustrate how latent recovery quality varies across easy, hard, semantically good, and semantically bad cases.}
  \label{fig:semantic_good_spatial_bad_grid}
\end{figure}
```

### spatial_good_semantic_bad_grid

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{image_grids/pdf/spatial_good_semantic_bad_grid.pdf}
  \caption{Selected generated-image examples illustrate how latent recovery quality varies across easy, hard, semantically good, and semantically bad cases.}
  \label{fig:spatial_good_semantic_bad_grid}
\end{figure}
```

### spatial_error_maps_best

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/pdf/spatial_error_maps_best.pdf}
  \caption{Spatial maps show token-wise error structure across the latent grid, revealing whether recovery failures are diffuse or localized.}
  \label{fig:spatial_error_maps_best}
\end{figure}
```

### spatial_error_maps_worst

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/pdf/spatial_error_maps_worst.pdf}
  \caption{Spatial maps show token-wise error structure across the latent grid, revealing whether recovery failures are diffuse or localized.}
  \label{fig:spatial_error_maps_worst}
\end{figure}
```

### spatial_error_maps_random

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/pdf/spatial_error_maps_random.pdf}
  \caption{Spatial maps show token-wise error structure across the latent grid, revealing whether recovery failures are diffuse or localized.}
  \label{fig:spatial_error_maps_random}
\end{figure}
```

### mean_spatial_l2_error_atlas

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/aggregate/pdf/mean_spatial_l2_error_atlas.pdf}
  \caption{Aggregate spatial atlas averaged over the full 10k set.}
  \label{fig:mean_spatial_l2_error_atlas}
\end{figure}
```

### mean_spatial_cosine_similarity_atlas

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/aggregate/pdf/mean_spatial_cosine_similarity_atlas.pdf}
  \caption{Aggregate spatial atlas averaged over the full 10k set.}
  \label{fig:mean_spatial_cosine_similarity_atlas}
\end{figure}
```

### mean_spatial_norm_difference_atlas

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{spatial_maps/aggregate/pdf/mean_spatial_norm_difference_atlas.pdf}
  \caption{Aggregate spatial atlas averaged over the full 10k set.}
  \label{fig:mean_spatial_norm_difference_atlas}
\end{figure}
```

### class_conditional_ratio_histogram

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/class_conditional_ratio_histogram.pdf}
  \caption{When labels are available, the class-level recovery ratios show whether some categories are systematically easier or harder to recover.}
  \label{fig:class_conditional_ratio_histogram}
\end{figure}
```

### best_classes_by_recovery_ratio

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/best_classes_by_recovery_ratio.pdf}
  \caption{Class-level ranking of recovery ratios can expose semantic categories that are especially easy or hard for the decoder/interface loop.}
  \label{fig:best_classes_by_recovery_ratio}
\end{figure}
```

### worst_classes_by_recovery_ratio

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{plots/pdf/worst_classes_by_recovery_ratio.pdf}
  \caption{Class-level ranking of recovery ratios can expose semantic categories that are especially easy or hard for the decoder/interface loop.}
  \label{fig:worst_classes_by_recovery_ratio}
\end{figure}
```
