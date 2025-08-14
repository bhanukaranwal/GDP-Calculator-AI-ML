// Performance tests (continued)
describe('Dashboard Performance', () => {
  test('renders within acceptable time', async () => {
    const startTime = performance.now();

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('GDP Analytics Dashboard')).toBeInTheDocument();
    });

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render within 2 seconds
    expect(renderTime).toBeLessThan(2000);
  });

  test('handles large datasets efficiently', async () => {
    // Mock large dataset
    const largeDataset = {
      ...mockDashboardData,
      recentCalculations: Array.from({ length: 1000 }, (_, i) => ({
        id: i.toString(),
        country_code: `C${i.toString().padStart(3, '0')}`,
        period: '2024-Q1',
        gdp_value: Math.random() * 50000,
        created_at: new Date().toISOString()
      }))
    };

    mockDashboardService.getDashboardData.mockResolvedValue(largeDataset);

    const startTime = performance.now();

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('GDP Analytics Dashboard')).toBeInTheDocument();
    });

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should still render efficiently with large dataset
    expect(renderTime).toBeLessThan(5000);
  });

  test('memory usage remains stable', async () => {
    const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;

    // Render and unmount multiple times
    for (let i = 0; i < 10; i++) {
      const { unmount } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('GDP Analytics Dashboard')).toBeInTheDocument();
      });

      unmount();
    }

    const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
    const memoryIncrease = finalMemory - initialMemory;

    // Memory increase should be reasonable (less than 50MB)
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
  });
});