"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var column_1 = require("./column");
var WidgetBoxView = /** @class */ (function (_super) {
    tslib_1.__extends(WidgetBoxView, _super);
    function WidgetBoxView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    WidgetBoxView.__name__ = "WidgetBoxView";
    return WidgetBoxView;
}(column_1.ColumnView));
exports.WidgetBoxView = WidgetBoxView;
var WidgetBox = /** @class */ (function (_super) {
    tslib_1.__extends(WidgetBox, _super);
    function WidgetBox(attrs) {
        return _super.call(this, attrs) || this;
    }
    WidgetBox.initClass = function () {
        this.prototype.default_view = WidgetBoxView;
    };
    WidgetBox.__name__ = "WidgetBox";
    return WidgetBox;
}(column_1.Column));
exports.WidgetBox = WidgetBox;
WidgetBox.initClass();
