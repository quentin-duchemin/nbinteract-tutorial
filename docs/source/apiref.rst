.. currentmodule:: wakapy

====================
Api reference
====================

|

.. note::

    The data containers are divided in two sections:
        1. Containers: They contain data and have some functionalities
           and processing properties.
        2. Containers-bare: They do not have any special
           functionality other than being datacontainers.

Containers
----------
Main classes, data containers with functionalities.
|

JsonDict
________
.. autoclass:: wakapy.parser.JsonDict
    :members:

User
____
.. autoclass:: wakapy.User
    :members:

Day
___
.. autoclass:: wakapy.day.Day
    :members:

Slice
_____
.. autoclass:: wakapy.user.Slice
    :members:


Containers-bare
---------------
The following classes are the static data containers


Parent
______
.. autoclass:: wakapy.containers.Parent
    :members:

Project
_______
.. autoclass:: wakapy.containers.Project
    :members:

Child Containers
________________
The following classes are all child's of :class:`.Parent`

.. autoclass:: wakapy.containers.Os
    :members:

.. autoclass:: wakapy.containers.Language
    :members:

.. autoclass:: wakapy.containers.Entity
    :members:

.. autoclass:: wakapy.containers.Dependency
    :members:

.. autoclass:: wakapy.containers.Category
    :members:

.. autoclass:: wakapy.containers.Editor
    :members:

.. autoclass:: wakapy.containers.Branch
    :members:

.. autoclass:: wakapy.containers.GrandTotal
    :members:

Extra
-----

PieChart
________

.. autoclass:: wakapy.plot.PieChart
    :members:

Utils
_____
.. automethod:: wakapy.utils.order_dict