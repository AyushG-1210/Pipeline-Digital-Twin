import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { PINNOutput } from '../../types';

interface IntegrityChartProps {
    pinnData: PINNOutput;
}

/**
 * Dual-line time series chart showing historical and predicted integrity
 */
export default function IntegrityChart({ pinnData }: IntegrityChartProps) {
    // Combine historical and predicted data
    const chartData = [
        ...pinnData.historical_integrity.map(point => ({
            time: new Date(point.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            historical: point.value,
            predicted: null,
            fullDate: point.time
        })),
        ...pinnData.predicted_integrity.map(point => ({
            time: new Date(point.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            historical: null,
            predicted: point.value,
            fullDate: point.time
        }))
    ];

    // Find the transition point (today)
    const todayIndex = pinnData.historical_integrity.length - 1;
    const todayDate = chartData[todayIndex]?.time || '';

    return (
        <div style={{ background: 'rgb(var(--color-surface))', borderRadius: '0.5rem' }}>
        <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-primary))" strokeOpacity={0.08} />
                <XAxis
                    dataKey="time"
                    stroke="rgb(var(--color-ink-muted))"
                    style={{ fontSize: '12px' }}
                />
                <YAxis
                    stroke="rgb(var(--color-ink-muted))"
                    domain={[0, 100]}
                    label={{ value: 'Integrity (%)', angle: -90, position: 'insideLeft', style: { fill: 'rgb(var(--color-ink-muted))' } }}
                    style={{ fontSize: '12px' }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'rgb(var(--color-surface))',
                        border: '1px solid rgb(var(--color-line))',
                        borderRadius: '8px',
                        color: 'rgb(var(--color-ink))'
                    }}
                    formatter={(value: any) => [`${value?.toFixed(1)}%`, '']}
                />
                <Legend
                    wrapperStyle={{ paddingTop: '20px', fontSize: '12px' }}
                    iconType="line"
                />

                {/* Today marker */}
                <ReferenceLine
                    x={todayDate}
                    stroke="rgb(var(--color-ink-muted))"
                    strokeDasharray="3 3"
                    label={{ value: 'Today', position: 'top', fill: 'rgb(var(--color-ink-muted))', fontSize: 12 }}
                />

                {/* Historical data line */}
                <Line
                    type="monotone"
                    dataKey="historical"
                    stroke="rgb(var(--color-primary))"
                    strokeWidth={2}
                    dot={{ fill: 'rgb(var(--color-primary))', r: 3 }}
                    name="Historical Sensor Integrity"
                    connectNulls={false}
                />

                {/* Predicted data line (dashed) */}
                <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#68B0AB"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ fill: '#68B0AB', r: 3 }}
                    name="Physics-Constrained Prediction"
                    connectNulls={false}
                />
            </LineChart>
        </ResponsiveContainer>
        </div>
    );
}
