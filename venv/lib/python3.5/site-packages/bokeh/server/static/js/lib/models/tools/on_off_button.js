"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var button_tool_1 = require("./button_tool");
var mixins_1 = require("../../styles/mixins");
var OnOffButtonView = /** @class */ (function (_super) {
    tslib_1.__extends(OnOffButtonView, _super);
    function OnOffButtonView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    OnOffButtonView.prototype.render = function () {
        _super.prototype.render.call(this);
        if (this.model.active)
            this.el.classList.add(mixins_1.bk_active);
        else
            this.el.classList.remove(mixins_1.bk_active);
    };
    OnOffButtonView.prototype._clicked = function () {
        var active = this.model.active;
        this.model.active = !active;
    };
    OnOffButtonView.__name__ = "OnOffButtonView";
    return OnOffButtonView;
}(button_tool_1.ButtonToolButtonView));
exports.OnOffButtonView = OnOffButtonView;
