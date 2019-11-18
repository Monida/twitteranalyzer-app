"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var action_tool_1 = require("./action_tool");
var icons_1 = require("../../../styles/icons");
var ResetToolView = /** @class */ (function (_super) {
    tslib_1.__extends(ResetToolView, _super);
    function ResetToolView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ResetToolView.prototype.doit = function () {
        this.plot_view.reset();
    };
    ResetToolView.__name__ = "ResetToolView";
    return ResetToolView;
}(action_tool_1.ActionToolView));
exports.ResetToolView = ResetToolView;
var ResetTool = /** @class */ (function (_super) {
    tslib_1.__extends(ResetTool, _super);
    function ResetTool(attrs) {
        var _this = _super.call(this, attrs) || this;
        _this.tool_name = "Reset";
        _this.icon = icons_1.bk_tool_icon_reset;
        return _this;
    }
    ResetTool.initClass = function () {
        this.prototype.default_view = ResetToolView;
    };
    ResetTool.__name__ = "ResetTool";
    return ResetTool;
}(action_tool_1.ActionTool));
exports.ResetTool = ResetTool;
ResetTool.initClass();
