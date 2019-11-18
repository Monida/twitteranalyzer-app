"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Settings = /** @class */ (function () {
    function Settings() {
        this._dev = false;
    }
    Object.defineProperty(Settings.prototype, "dev", {
        get: function () {
            return this._dev;
        },
        set: function (dev) {
            this._dev = dev;
        },
        enumerable: true,
        configurable: true
    });
    Settings.__name__ = "Settings";
    return Settings;
}());
exports.Settings = Settings;
exports.settings = new Settings();
