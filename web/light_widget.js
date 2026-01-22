import { app } from "../../scripts/app.js";

// Light Widget for LightForge3DGS
// Visual representation of light azimuth and elevation

app.registerExtension({
    name: "LightForge3DGS.LightWidget",

    async nodeCreated(node) {
        if (node.comfyClass !== "Relight3DGS") return;

        // Find the widgets we need to visualize
        const azimuthWidget = node.widgets?.find(w => w.name === "azimuth");
        const elevationWidget = node.widgets?.find(w => w.name === "elevation");
        const intensityWidget = node.widgets?.find(w => w.name === "intensity");
        const temperatureWidget = node.widgets?.find(w => w.name === "temperature");

        if (!azimuthWidget || !elevationWidget) return;

        // Create canvas widget
        const canvasWidget = node.addDOMWidget("light_preview", "canvas", document.createElement("div"), {
            serialize: false,
        });

        const container = canvasWidget.element;
        container.style.width = "100%";
        container.style.height = "200px";
        container.style.position = "relative";
        container.style.overflow = "hidden";

        const canvas = document.createElement("canvas");
        canvas.width = 300;
        canvas.height = 200;
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        canvas.style.borderRadius = "8px";
        container.appendChild(canvas);

        const ctx = canvas.getContext("2d");

        // Info display
        const infoDiv = document.createElement("div");
        infoDiv.style.cssText = `
            position: absolute;
            bottom: 5px;
            left: 0;
            right: 0;
            display: flex;
            justify-content: space-around;
            font-size: 11px;
            color: #00ffff;
            font-family: monospace;
        `;
        container.appendChild(infoDiv);

        function getDirectionName(azimuth) {
            if (azimuth >= 337.5 || azimuth < 22.5) return "NORTH";
            if (azimuth >= 22.5 && azimuth < 67.5) return "NE";
            if (azimuth >= 67.5 && azimuth < 112.5) return "EAST";
            if (azimuth >= 112.5 && azimuth < 157.5) return "SE";
            if (azimuth >= 157.5 && azimuth < 202.5) return "SOUTH";
            if (azimuth >= 202.5 && azimuth < 247.5) return "SW";
            if (azimuth >= 247.5 && azimuth < 292.5) return "WEST";
            return "NW";
        }

        function getTempColor(temp) {
            if (temp > 0.3) return "#ffaa44"; // warm
            if (temp < -0.3) return "#44aaff"; // cool
            return "#ffffff"; // neutral
        }

        function draw() {
            const az = azimuthWidget.value;
            const el = elevationWidget.value;
            const intensity = intensityWidget?.value || 1.0;
            const temp = temperatureWidget?.value || 0.0;

            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2 + 20;
            const radius = Math.min(w, h) * 0.35;

            // Clear
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, w, h);

            // Draw grid (ellipse for perspective)
            ctx.strokeStyle = "#333355";
            ctx.lineWidth = 1;

            // Horizontal ellipse (ground)
            ctx.beginPath();
            ctx.ellipse(cx, cy, radius, radius * 0.4, 0, 0, Math.PI * 2);
            ctx.stroke();

            // Cross lines on grid
            ctx.beginPath();
            ctx.moveTo(cx - radius, cy);
            ctx.lineTo(cx + radius, cy);
            ctx.moveTo(cx, cy - radius * 0.4);
            ctx.lineTo(cx, cy + radius * 0.4);
            ctx.stroke();

            // Draw azimuth arc
            ctx.strokeStyle = "#ff66aa";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.ellipse(cx, cy, radius * 0.8, radius * 0.32, 0, 0, Math.PI * 2);
            ctx.stroke();

            // Calculate light position
            const azRad = (az - 90) * Math.PI / 180; // -90 to align 0 = North
            const elRad = el * Math.PI / 180;

            // Project to 2D (isometric-ish)
            const lightX = cx + Math.cos(azRad) * radius * 0.8;
            const lightY = cy - Math.sin(azRad) * radius * 0.32 - Math.sin(elRad) * radius * 0.6;

            // Draw elevation arc
            ctx.strokeStyle = "#44ffaa";
            ctx.lineWidth = 2;
            ctx.beginPath();
            const arcStartX = cx + Math.cos(azRad) * radius * 0.8;
            const arcStartY = cy - Math.sin(azRad) * radius * 0.32;
            ctx.moveTo(arcStartX, arcStartY);
            ctx.quadraticCurveTo(
                arcStartX,
                arcStartY - radius * 0.4,
                lightX,
                lightY
            );
            ctx.stroke();

            // Draw line from center to light
            ctx.strokeStyle = "#666688";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(lightX, lightY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw center object (small cube representation)
            ctx.fillStyle = "#888899";
            ctx.fillRect(cx - 8, cy - 12, 16, 16);
            ctx.strokeStyle = "#aaaacc";
            ctx.strokeRect(cx - 8, cy - 12, 16, 16);

            // Draw light source (sun/lamp)
            const lightColor = getTempColor(temp);
            const lightRadius = 8 + intensity * 4;

            // Glow
            const gradient = ctx.createRadialGradient(lightX, lightY, 0, lightX, lightY, lightRadius * 2);
            gradient.addColorStop(0, lightColor);
            gradient.addColorStop(1, "transparent");
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(lightX, lightY, lightRadius * 2, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.fillStyle = lightColor;
            ctx.beginPath();
            ctx.arc(lightX, lightY, lightRadius, 0, Math.PI * 2);
            ctx.fill();

            // Update info
            infoDiv.innerHTML = `
                <span style="color:#ff66aa">AZ: ${az.toFixed(0)}° (${getDirectionName(az)})</span>
                <span style="color:#44ffaa">EL: ${el.toFixed(0)}°</span>
                <span style="color:${lightColor}">INT: ${intensity.toFixed(1)}</span>
            `;
        }

        // Initial draw
        draw();

        // Redraw on widget changes
        const originalAzCallback = azimuthWidget.callback;
        azimuthWidget.callback = function (v) {
            if (originalAzCallback) originalAzCallback.call(this, v);
            draw();
        };

        const originalElCallback = elevationWidget.callback;
        elevationWidget.callback = function (v) {
            if (originalElCallback) originalElCallback.call(this, v);
            draw();
        };

        if (intensityWidget) {
            const originalIntCallback = intensityWidget.callback;
            intensityWidget.callback = function (v) {
                if (originalIntCallback) originalIntCallback.call(this, v);
                draw();
            };
        }

        if (temperatureWidget) {
            const originalTempCallback = temperatureWidget.callback;
            temperatureWidget.callback = function (v) {
                if (originalTempCallback) originalTempCallback.call(this, v);
                draw();
            };
        }

        // Force redraw on node resize
        node.onResize = function () {
            draw();
        };
    }
});
