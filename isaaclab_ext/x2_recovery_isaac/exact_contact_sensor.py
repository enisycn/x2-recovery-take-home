"""Exact-path PhysX contact sensor for hierarchical URDF-converted links."""

from __future__ import annotations

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors import ContactSensor


class ExactPathContactSensor(ContactSensor):
    """Create a one-body PhysX view directly from the configured full path.

    Isaac Lab's generic contact sensor rebuilds a path from a parent and leaf
    name.  That is useful for flat link layouts but duplicates the final path
    component for the hierarchical X2 USD.  This small adapter keeps all data
    buffers and update logic from the official sensor and only fixes view setup.
    """

    def _initialize_impl(self) -> None:
        BaseContactSensor._initialize_impl(self)
        self._physics_sim_view = PhysxManager.get_physics_sim_view()

        body_glob = self.cfg.prim_path.replace(".*", "*")
        filter_globs = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_glob)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_glob,
            filter_patterns=filter_globs,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * self._num_envs,
        )
        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != 1:
            raise RuntimeError(
                "Exact-path contact sensor must resolve one body per environment."
                f"\n\tInput prim path: {self.cfg.prim_path}"
                f"\n\tResolved count : {self.body_physx_view.count}"
            )
        self._create_buffers()
