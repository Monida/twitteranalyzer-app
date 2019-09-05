"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var action_tool_1 = require("./action_tool");
var p = require("../../../core/properties");
var toolbar_1 = require("../../../styles/toolbar");
var CustomActionButtonView = /** @class */ (function (_super) {
    tslib_1.__extends(CustomActionButtonView, _super);
    function CustomActionButtonView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    CustomActionButtonView.prototype.css_classes = function () {
        return _super.prototype.css_classes.call(this).concat(toolbar_1.bk_toolbar_button_custom_action);
    };
    CustomActionButtonView.__name__ = "CustomActionButtonView";
    return CustomActionButtonView;
}(action_tool_1.ActionToolButtonView));
exports.CustomActionButtonView = CustomActionButtonView;
var CustomActionView = /** @class */ (function (_super) {
    tslib_1.__extends(CustomActionView, _super);
    function CustomActionView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    CustomActionView.prototype.doit = function () {
        if (this.model.callback != null)
            this.model.callback.execute(this.model);
    };
    CustomActionView.__name__ = "CustomActionView";
    return CustomActionView;
}(action_tool_1.ActionToolView));
exports.CustomActionView = CustomActionView;
var CustomAction = /** @class */ (function (_super) {
    tslib_1.__extends(CustomAction, _super);
    function CustomAction(attrs) {
        var _this = _super.call(this, attrs) || this;
        _this.tool_name = "Custom Action";
        _this.button_view = CustomActionButtonView;
        return _this;
    }
    CustomAction.initClass = function () {
        this.prototype.default_view = CustomActionView;
        this.define({
            action_tooltip: [p.String, 'Perform a Custom Action'],
            callback: [p.Any],
            icon: [p.String,],
        });
    };
    Object.defineProperty(CustomAction.prototype, "tooltip", {
        get: function () {
            return this.action_tooltip;
        },
        enumerable: true,
        configurable: true
    });
    CustomAction.__name__ = "CustomAction";
    return CustomAction;
}(action_tool_1.ActionTool));
exports.CustomAction = CustomAction;
CustomAction.initClass();
