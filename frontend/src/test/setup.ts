// jsdom does no layout, so it has no scrolling.
Element.prototype.scrollIntoView = function scrollIntoView() {};
