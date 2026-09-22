# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch


# =========================
# Projection Vector Generation
# =========================
class Projections:
    @staticmethod
    def generate_random_projections(num_projections, D, device=None, dtype=None):
        """
        Generates a set of random, normalized projection vectors on the unit sphere.
        """
        P_directions = torch.randn(num_projections, D, device=device, dtype=dtype)
        P_directions = P_directions / torch.norm(P_directions, dim=1, keepdim=True)
        return P_directions

    @staticmethod
    def generate_svd_projections_from_features(z):
        """
        Computes right-singular vectors (V^T) of a centered feature matrix.
        Attempts torch.linalg.svd first, falling back to LOBPCG/eigh if needed.
        """
        with torch.amp.autocast('cuda', enabled=False):
            z = z.detach().float()
            z_centered = z - z.mean(dim=0)

            try:
                _, _, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                return Vt
            except Exception:
                B, D = z_centered.shape
                k = min(B, D)
                A = torch.matmul(z_centered.T, z_centered)
                X = torch.randn(D, k, device=z.device, dtype=torch.float32)

                try:
                    _, V = torch.lobpcg(A, X=X, largest=True)
                    Vt = V.T
                except Exception:
                    _, V = torch.linalg.eigh(A)
                    Vt = V.T.flip(0)[:k]
                return Vt

    @staticmethod
    def generate_svd_projections(z1, z2):
        """
        Computes the right-singular vectors (V^T) of the centered feature matrices z1 and z2.
        Attempts torch.linalg.svd first, falling back to LOBPCG if it fails.
        """
        Vt_z1 = Projections.generate_svd_projections_from_features(z1)
        Vt_z2 = Projections.generate_svd_projections_from_features(z2)
        return Vt_z1, Vt_z2
    

    @staticmethod
    def get_projection_vectors(
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_projections: int,
        projection_vectors_type: str,
        proj_output_dim: int,
    ):
        """
        Main entry point to get projection vectors based on the specified type.
        Supports: 'random', 'torch_svd_and_random', 'torch_svd_bottom_half_eigen_and_random'.
        """
        if projection_vectors_type == 'torch_svd_and_random':
            # Combine top eigenvectors with random projections
            Vt_z1, Vt_z2 = Projections.generate_svd_projections(z1, z2)
            random_projs = Projections.generate_random_projections(
                num_projections - Vt_z1.size(0), proj_output_dim, device=z1.device, dtype=z1.dtype
            )
            return [torch.vstack([Vt_z1, random_projs]), torch.vstack([Vt_z2, random_projs])]

        elif projection_vectors_type == 'torch_svd_bottom_half_eigen_and_random':
            # Combine bottom-half eigenvectors with random projections
            Vt_z1, Vt_z2 = Projections.generate_svd_projections(z1, z2)
            Vt_z1_bh = Vt_z1[Vt_z1.size(0)//2:]
            Vt_z2_bh = Vt_z2[Vt_z2.size(0)//2:]
            random_projs = Projections.generate_random_projections(
                num_projections - Vt_z1_bh.size(0), proj_output_dim, device=z1.device, dtype=z1.dtype
            )
            return [torch.vstack([Vt_z1_bh, random_projs]), torch.vstack([Vt_z2_bh, random_projs])]

        elif projection_vectors_type == 'random':
            # Purely random projections
            return Projections.generate_random_projections(
                num_projections, proj_output_dim, device=z1.device, dtype=z1.dtype
            )
        else:
            raise ValueError(f"Unsupported projection_vectors_type: {projection_vectors_type}")

    @staticmethod
    def get_shared_projection_vectors(
        z: torch.Tensor,
        num_projections: int,
        projection_vectors_type: str,
        proj_output_dim: int,
    ):
        """
        Returns a single projection matrix shared across multiple views.
        For SVD-based options, eigenvectors are extracted from a shared feature set.
        """
        if projection_vectors_type == 'random':
            return Projections.generate_random_projections(
                num_projections, proj_output_dim, device=z.device, dtype=z.dtype
            )

        Vt = Projections.generate_svd_projections_from_features(z)
        if projection_vectors_type == 'torch_svd_bottom_half_eigen_and_random':
            Vt = Vt[Vt.size(0) // 2:]
        elif projection_vectors_type != 'torch_svd_and_random':
            raise ValueError(f"Unsupported projection_vectors_type: {projection_vectors_type}")

        random_count = max(0, num_projections - Vt.size(0))
        if random_count == 0:
            return Vt[:num_projections]

        random_projs = Projections.generate_random_projections(
            random_count, proj_output_dim, device=z.device, dtype=z.dtype
        )
        return torch.vstack([Vt, random_projs])
