# HARP Artifact Assets

The source and evaluation instructions are in the repository. Download the assets below for DiT FID and Wan VBench. The six-model performance workflow runs from the source checkout.

Save the three `docker-images.tar.gz.part-*` files and `docker-images.tar.gz.sha256` in `containers/` at the repository root. For DiT, save `dit-fid.tar.gz` and `dit-fid.tar.gz.sha256` in `scoring-assets/`. For Wan, save the five `wan-vbench.tar.gz.part-*` files and `wan-vbench.tar.gz.sha256` in `scoring-assets/`. Create these directories if needed.

```bash
# From the repository root, for both quality workflows
(cd containers && cat docker-images.tar.gz.part-{01..03} > docker-images.tar.gz &&
  sha256sum -c docker-images.tar.gz.sha256)

# For DiT FID
(cd scoring-assets && sha256sum -c dit-fid.tar.gz.sha256)

# For Wan VBench
(cd scoring-assets && cat wan-vbench.tar.gz.part-{01..05} > wan-vbench.tar.gz &&
  sha256sum -c wan-vbench.tar.gz.sha256)
```

Continue with the [repository README](https://github.com/YudongMu/HARP-Artifact/blob/main/README.md#2-load-the-images-and-check-gpu-access).
