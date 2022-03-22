from __future__ import annotations
import abc
import sys

import maya.api.OpenMaya as om


class BaseNode(om.MPxNode):
    """
    base node for all nodes
    """
    def __init__(self) -> None:
        """
        initializes the node
        """
        om.MPxNode.__init__(self)

    def create(self) -> BaseNode:
        """
        creates the node
        """
        return type(self)()

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def compute(self, plug: om.MPlug, block: om.MDataBlock) -> None:
        pass

    def register_node(self, plugin: om.MFnPlugin, name: str, id_: om.MTypeId) -> None:
        """
        registers this node
        """
        try:
            plugin.registerNode(name, id_, self.create, self.initialize)
        except:
            sys.stderr.write("Failed to register plugin")
            raise

    @staticmethod
    def deregister_node(plugin: om.MFnPlugin, id_: om.MTypeId) -> None:
        """
        deregisters this node
        """
        try:
            plugin.deregisterNode(id_)
        except:
            sys.stderr.write("Failed to deregister plugin")
            raise


class BaseCommand(om.MPxCommand):
    """
    base command for all commands
    """
    def __init__(self):
        """
        initializes the command
        """
        om.MPxCommand.__init__(self)

    def create(self) -> BaseCommand:
        """
        creates the command
        """
        return type(self)()

    @abc.abstractmethod
    def create_syntax(self) -> om.MSyntax:
        pass

    @abc.abstractmethod
    def parse_arguments(self, arguments: om.MArgList) -> None:
        pass

    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def doIt(self, arguments: om.MArgList) -> None:
        pass

    def register_command(self, plugin: om.MFnPlugin, name: str) -> None:
        """
        registers this command
        """
        try:
            plugin.registerCommand(name, self.create, self.create_syntax)
        except:
            sys.stderr.write("Failed to register plugin")
            raise

    @staticmethod
    def deregister_command(plugin: om.MFnPlugin, name: str) -> None:
        """
        deregisters this command
        """
        try:
            plugin.deregisterNode(name)
        except:
            sys.stderr.write("Failed to deregister plugin")
            raise
