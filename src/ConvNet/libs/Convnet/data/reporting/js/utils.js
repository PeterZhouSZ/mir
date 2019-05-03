function spaceToText(space)
{
	if (space > (4 * Math.pow(2,40)))
		return Number(space / Math.pow(2,40)).toFixed(1) +"TB";
	if (space > (4 * Math.pow(2,30)))
		return Number(space / Math.pow(2,30)).toFixed(1) +"GB";
	if (space > (4 * Math.pow(2,20)))
		return Number(space / Math.pow(2,20)).toFixed(1) +"MB";
	if (space > (4 * Math.pow(2,10)))
		return Number(space / Math.pow(2,10)).toFixed(1) +"KB";
	return space + "B";
}



function drawSpaceChart(total, free, canvas)
{
	var ctx = canvas.getContext("2d");
	ctx.fillStyle = "#A0A0FF";
	ctx.strokeStyle = "#D0D0D0";
	ctx.lineWidth=2;
	ctx.clearRect(0, 0, 128, 145);

	var percentage = 1 - free / total;

	ctx.beginPath();
	ctx.arc(64, 64, 60, Math.PI*3/2 + 0, Math.PI*3/2 + Math.PI*2*percentage);
	ctx.arc(64, 64, 45, Math.PI*3/2 + Math.PI*2*percentage, Math.PI*3/2 + 0, true);
	ctx.fill();

	ctx.fillStyle = "#A0A0A0";
	ctx.font="20px Arial";
	var text = Number(percentage*100).toFixed(1)+"%";
	var textMeasure = ctx.measureText(text);
	ctx.textBaseline="middle";
	ctx.fillText(text, 64 - textMeasure.width/2, 64);

	ctx.font="13px Arial";
	text = spaceToText(free) + " / " + spaceToText(total);
	textMeasure = ctx.measureText(text);
	ctx.textBaseline="bottom";
	ctx.fillText(text, 64 - textMeasure.width/2, 145);


	ctx.beginPath();
	ctx.arc(64, 64, 60, 0, 2*Math.PI);
	ctx.stroke();
	ctx.beginPath();
	ctx.arc(64, 64, 45, 0, 2*Math.PI);
	ctx.stroke();
}

function drawSpaceChartFail(msg, canvas)
{
	var ctx = canvas.getContext("2d");
	ctx.fillStyle = "#A0A0FF";
	ctx.strokeStyle = "#D0D0D0";
	ctx.lineWidth=2;
	ctx.clearRect(0, 0, 128, 145);


	ctx.fillStyle = "#A0A0A0";
	ctx.font="20px Arial";
	var text = "N/A";
	var textMeasure = ctx.measureText(text);
	ctx.textBaseline="middle";
	ctx.fillText(text, 64 - textMeasure.width/2, 64);

	ctx.font="13px Arial";
	text = msg;
	textMeasure = ctx.measureText(text);
	ctx.textBaseline="bottom";
	ctx.fillText(text, 64 - textMeasure.width/2, 145);

	ctx.beginPath();
	ctx.arc(64, 64, 60, 0, 2*Math.PI);
	ctx.stroke();
	ctx.beginPath();
	ctx.arc(64, 64, 45, 0, 2*Math.PI);
	ctx.stroke();
}

function initializeSpaceChart(canvas)
{
	canvas.width=128;
	canvas.height=145;
	drawSpaceChartFail("Refresh pending", canvas);
}

