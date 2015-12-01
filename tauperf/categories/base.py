from rootpy.tree import Cut


class CategoryMeta(type):
    """
    Metaclass for all categories
    """
    CATEGORY_REGISTRY = {}
    def __new__(cls, name, bases, dct):
        if name in CategoryMeta.CATEGORY_REGISTRY:
            raise ValueError("Multiple categories with the same name: %s" % name)
        cat = type.__new__(cls, name, bases, dct)
        # register the category
        CategoryMeta.CATEGORY_REGISTRY[name] = cat
        return cat


class Category(object):

    __metaclass__ = CategoryMeta

    cuts = Cut()
    common_cuts = Cut()
    plot_label = None

    @classmethod
    def get_cuts(cls):
        cuts = cls.cuts & cls.common_cuts
        return cuts

    @classmethod
    def get_parent(cls):
        return cls

