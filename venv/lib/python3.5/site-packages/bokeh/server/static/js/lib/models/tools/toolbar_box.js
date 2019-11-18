"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var p = require("../../core/properties");
var array_1 = require("../../core/util/array");
var toolbar_base_1 = require("./toolbar_base");
var tool_proxy_1 = require("./tool_proxy");
var layout_dom_1 = require("../layouts/layout_dom");
var layout_1 = require("../../core/layout");
var ProxyToolbar = /** @class */ (function (_super) {
    tslib_1.__extends(ProxyToolbar, _super);
    function ProxyToolbar(attrs) {
        return _super.call(this, attrs) || this;
    }
    ProxyToolbar.prototype.initialize = function () {
        _super.prototype.initialize.call(this);
        this._merge_tools();
    };
    ProxyToolbar.prototype._merge_tools = function () {
        var _a;
        var _this = this;
        // Go through all the tools on the toolbar and replace them with
        // a proxy e.g. PanTool, BoxSelectTool, etc.
        this._proxied_tools = [];
        var inspectors = {};
        var actions = {};
        var gestures = {};
        var new_help_tools = [];
        var new_help_urls = [];
        for (var _i = 0, _b = this.help; _i < _b.length; _i++) {
            var helptool = _b[_i];
            if (!array_1.includes(new_help_urls, helptool.redirect)) {
                new_help_tools.push(helptool);
                new_help_urls.push(helptool.redirect);
            }
        }
        (_a = this._proxied_tools).push.apply(_a, new_help_tools);
        this.help = new_help_tools;
        for (var event_type in this.gestures) {
            var gesture = this.gestures[event_type];
            if (!(event_type in gestures)) {
                gestures[event_type] = {};
            }
            for (var _c = 0, _d = gesture.tools; _c < _d.length; _c++) {
                var tool = _d[_c];
                if (!(tool.type in gestures[event_type])) {
                    gestures[event_type][tool.type] = [];
                }
                gestures[event_type][tool.type].push(tool);
            }
        }
        for (var _e = 0, _f = this.inspectors; _e < _f.length; _e++) {
            var tool = _f[_e];
            if (!(tool.type in inspectors)) {
                inspectors[tool.type] = [];
            }
            inspectors[tool.type].push(tool);
        }
        for (var _g = 0, _h = this.actions; _g < _h.length; _g++) {
            var tool = _h[_g];
            if (!(tool.type in actions)) {
                actions[tool.type] = [];
            }
            actions[tool.type].push(tool);
        }
        // Add a proxy for each of the groups of tools.
        var make_proxy = function (tools, active) {
            if (active === void 0) { active = false; }
            var proxy = new tool_proxy_1.ToolProxy({ tools: tools, active: active });
            _this._proxied_tools.push(proxy);
            return proxy;
        };
        for (var event_type in gestures) {
            var gesture = this.gestures[event_type];
            gesture.tools = [];
            for (var tool_type in gestures[event_type]) {
                var tools = gestures[event_type][tool_type];
                if (tools.length > 0) {
                    if (event_type == 'multi') {
                        for (var _j = 0, tools_1 = tools; _j < tools_1.length; _j++) {
                            var tool = tools_1[_j];
                            var proxy = make_proxy([tool]);
                            gesture.tools.push(proxy);
                            this.connect(proxy.properties.active.change, this._active_change.bind(this, proxy));
                        }
                    }
                    else {
                        var proxy = make_proxy(tools);
                        gesture.tools.push(proxy);
                        this.connect(proxy.properties.active.change, this._active_change.bind(this, proxy));
                    }
                }
            }
        }
        this.actions = [];
        for (var tool_type in actions) {
            var tools = actions[tool_type];
            if (tool_type == 'CustomAction') {
                for (var _k = 0, tools_2 = tools; _k < tools_2.length; _k++) {
                    var tool = tools_2[_k];
                    this.actions.push(make_proxy([tool]));
                }
            }
            else if (tools.length > 0) {
                this.actions.push(make_proxy(tools)); // XXX
            }
        }
        this.inspectors = [];
        for (var tool_type in inspectors) {
            var tools = inspectors[tool_type];
            if (tools.length > 0)
                this.inspectors.push(make_proxy(tools, true)); // XXX
        }
        for (var et in this.gestures) {
            var gesture = this.gestures[et];
            if (gesture.tools.length == 0)
                continue;
            gesture.tools = array_1.sort_by(gesture.tools, function (tool) { return tool.default_order; });
            if (!(et == 'pinch' || et == 'scroll' || et == 'multi'))
                gesture.tools[0].active = true;
        }
    };
    ProxyToolbar.__name__ = "ProxyToolbar";
    return ProxyToolbar;
}(toolbar_base_1.ToolbarBase));
exports.ProxyToolbar = ProxyToolbar;
var ToolbarBoxView = /** @class */ (function (_super) {
    tslib_1.__extends(ToolbarBoxView, _super);
    function ToolbarBoxView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ToolbarBoxView.prototype.initialize = function () {
        this.model.toolbar.toolbar_location = this.model.toolbar_location;
        _super.prototype.initialize.call(this);
    };
    Object.defineProperty(ToolbarBoxView.prototype, "child_models", {
        get: function () {
            return [this.model.toolbar]; // XXX
        },
        enumerable: true,
        configurable: true
    });
    ToolbarBoxView.prototype._update_layout = function () {
        this.layout = new layout_1.ContentBox(this.child_views[0].el);
        var toolbar = this.model.toolbar;
        if (toolbar.horizontal) {
            this.layout.set_sizing({
                width_policy: "fit", min_width: 100, height_policy: "fixed",
            });
        }
        else {
            this.layout.set_sizing({
                width_policy: "fixed", height_policy: "fit", min_height: 100,
            });
        }
    };
    ToolbarBoxView.__name__ = "ToolbarBoxView";
    return ToolbarBoxView;
}(layout_dom_1.LayoutDOMView));
exports.ToolbarBoxView = ToolbarBoxView;
var ToolbarBox = /** @class */ (function (_super) {
    tslib_1.__extends(ToolbarBox, _super);
    function ToolbarBox(attrs) {
        return _super.call(this, attrs) || this;
    }
    ToolbarBox.initClass = function () {
        this.prototype.default_view = ToolbarBoxView;
        this.define({
            toolbar: [p.Instance],
            toolbar_location: [p.Location, "right"],
        });
    };
    ToolbarBox.__name__ = "ToolbarBox";
    return ToolbarBox;
}(layout_dom_1.LayoutDOM));
exports.ToolbarBox = ToolbarBox;
ToolbarBox.initClass();
