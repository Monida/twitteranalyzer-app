"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var p = require("../../core/properties");
var types_1 = require("../../core/util/types");
var array_1 = require("../../core/util/array");
var inspect_tool_1 = require("./inspectors/inspect_tool");
var toolbar_base_1 = require("./toolbar_base");
var _get_active_attr = function (et) {
    switch (et) {
        case 'tap': return 'active_tap';
        case 'pan': return 'active_drag';
        case 'pinch':
        case 'scroll': return 'active_scroll';
        case 'multi': return 'active_multi';
    }
    return null;
};
var _supports_auto = function (et) {
    return et == 'tap' || et == 'pan';
};
var Toolbar = /** @class */ (function (_super) {
    tslib_1.__extends(Toolbar, _super);
    function Toolbar(attrs) {
        return _super.call(this, attrs) || this;
    }
    Toolbar.initClass = function () {
        this.prototype.default_view = toolbar_base_1.ToolbarBaseView;
        this.define({
            active_drag: [p.Any, 'auto'],
            active_inspect: [p.Any, 'auto'],
            active_scroll: [p.Any, 'auto'],
            active_tap: [p.Any, 'auto'],
            active_multi: [p.Any, null],
        });
    };
    Toolbar.prototype.connect_signals = function () {
        var _this = this;
        _super.prototype.connect_signals.call(this);
        this.connect(this.properties.tools.change, function () { return _this._init_tools(); });
    };
    Toolbar.prototype._init_tools = function () {
        var _this = this;
        _super.prototype._init_tools.call(this);
        if (this.active_inspect == 'auto') {
            // do nothing as all tools are active be default
        }
        else if (this.active_inspect instanceof inspect_tool_1.InspectTool) {
            var found = false;
            for (var _i = 0, _a = this.inspectors; _i < _a.length; _i++) {
                var inspector = _a[_i];
                if (inspector != this.active_inspect)
                    inspector.active = false;
                else
                    found = true;
            }
            if (!found) {
                this.active_inspect = null;
            }
        }
        else if (types_1.isArray(this.active_inspect)) {
            var active_inspect = array_1.intersection(this.active_inspect, this.inspectors);
            if (active_inspect.length != this.active_inspect.length) {
                this.active_inspect = active_inspect;
            }
            for (var _b = 0, _c = this.inspectors; _b < _c.length; _b++) {
                var inspector = _c[_b];
                if (!array_1.includes(this.active_inspect, inspector))
                    inspector.active = false;
            }
        }
        else if (this.active_inspect == null) {
            for (var _d = 0, _e = this.inspectors; _d < _e.length; _d++) {
                var inspector = _e[_d];
                inspector.active = false;
            }
        }
        var _activate_gesture = function (tool) {
            if (tool.active) {
                // tool was activated by a proxy, but we need to finish configuration manually
                _this._active_change(tool);
            }
            else
                tool.active = true;
        };
        // Connecting signals has to be done before changing the active state of the tools.
        for (var et in this.gestures) {
            var gesture = this.gestures[et];
            gesture.tools = array_1.sort_by(gesture.tools, function (tool) { return tool.default_order; });
            for (var _f = 0, _g = gesture.tools; _f < _g.length; _f++) {
                var tool = _g[_f];
                this.connect(tool.properties.active.change, this._active_change.bind(this, tool));
            }
        }
        for (var et in this.gestures) {
            var active_attr = _get_active_attr(et);
            if (active_attr) {
                var active_tool = this[active_attr];
                if (active_tool == 'auto') {
                    var gesture = this.gestures[et];
                    if (gesture.tools.length != 0 && _supports_auto(et)) {
                        _activate_gesture(gesture.tools[0]);
                    }
                }
                else if (active_tool != null) {
                    if (array_1.includes(this.tools, active_tool)) {
                        _activate_gesture(active_tool);
                    }
                    else {
                        this[active_attr] = null;
                    }
                }
            }
        }
    };
    Toolbar.__name__ = "Toolbar";
    return Toolbar;
}(toolbar_base_1.ToolbarBase));
exports.Toolbar = Toolbar;
Toolbar.initClass();
