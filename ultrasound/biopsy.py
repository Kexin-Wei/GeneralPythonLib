import vtk
from pathlib import Path
from enum import Enum


class CorePlanType(Enum):
    TEN = 10
    TWELVE = 12


class BiopsyPlanWithBox:
    def __init__(self, prostate_file: str, specimen_file: str, result_path: str):
        self.prostate_file = Path(prostate_file)
        self.specimen_file = Path(specimen_file)
        self.result_path = Path(result_path)

    def plan(self, plan_method: CorePlanType = CorePlanType.TWELVE):
        colors = vtk.vtkNamedColors()

        prostate_reader = vtk.vtkSTLReader()
        prostate_reader.SetFileName(str(self.prostate_file))
        prostate_reader.Update()

        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputData(prostate_reader.GetOutput())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetColor(colors.GetColor3d("Red"))
        actor1.GetProperty().SetOpacity(0.5)

        prostate_bounds = actor1.GetBounds()
        p1 = (prostate_bounds[1], prostate_bounds[2])
        p2 = (prostate_bounds[0], prostate_bounds[3])

        if plan_method == CorePlanType.TEN:
            core_points = self.ten_cores(p1, p2)
        else:
            core_points = self.twelve_cores(p1, p2)
        actors = []
        for p in core_points:
            actors.append(self.add_specimen(p))

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor1)
        for a in actors:
            renderer.AddActor(a)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        window.SetSize(800, 800)
        window.Render()

        window_to_img = vtk.vtkWindowToImageFilter()
        window_to_img.SetInput(window)
        window_to_img.SetInputBufferTypeToRGB()
        window_to_img.ReadFrontBufferOff()
        window_to_img.Update()

        prefix = "ten_core" if plan_method == CorePlanType.TEN else "twelve_core"
        png_file_name = self.result_path.joinpath(
            f"{prefix}_{self.prostate_file.stem}.png"
        )
        png_writer = vtk.vtkPNGWriter()
        png_writer.SetFileName(png_file_name)
        png_writer.SetInputConnection(window_to_img.GetOutputPort())
        png_writer.Write()

        interactor.Initialize()
        interactor.Start()

    def add_specimen(self, p: tuple):
        colors = vtk.vtkNamedColors()
        assert len(p) == 2, "p must be a tuple of length 2"
        x, y = p
        specimen_reader = vtk.vtkSTLReader()
        specimen_reader.SetFileName(str(self.specimen_file))

        transform = vtk.vtkTransform()
        transform.Translate(x, y, 0)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(specimen_reader.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(transform_filter.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Blue"))
        return actor

    @staticmethod
    def cude_center_of_two_points(p1: tuple, p3: tuple):
        assert len(p1) == 2 and len(p3) == 2, "p1 and p3 must be a tuple of length 2"
        x1, y1 = p1
        x2, y2 = p3
        return (x1 + x2) / 2, (y1 + y2) / 2

    def fourth_center_of_later(self, p1: tuple, p3: tuple):
        assert len(p1) == 2 and len(p3) == 2, "p1 and p3 must be a tuple of length 2"
        p_mid = self.cude_center_of_two_points(p1, p3)
        p_3_mid = self.cude_center_of_two_points(p3, p_mid)
        return p_3_mid

    def four_cude_center_from_two_points(self, p1: tuple, p3: tuple, left: bool = True):
        assert len(p1) == 2 and len(p3) == 2, "p1 and p3 must be a tuple of length 2"
        x1, y1 = p1
        x2, y2 = p3
        p2 = (x2, y1)
        p4 = (x1, y2)
        p_mid = self.cude_center_of_two_points(p1, p3)
        centers = []
        if left:
            centers.append(self.cude_center_of_two_points(p1, p_mid))
            centers.append(self.cude_center_of_two_points(p2, p_mid))
            centers.append(self.cude_center_of_two_points(p3, p_mid))
            centers.append(self.fourth_center_of_later(p4, p_mid))
        else:  # right
            centers.append(self.cude_center_of_two_points(p1, p_mid))
            centers.append(self.cude_center_of_two_points(p2, p_mid))
            centers.append(self.cude_center_of_two_points(p4, p_mid))
            centers.append(self.fourth_center_of_later(p3, p_mid))
        return centers

    def ten_cores(self, p1: tuple, p3: tuple):
        assert len(p1) == 2 and len(p3) == 2, "p1 and p3 must be a tuple of length 2"
        x1, y1 = p1
        x2, y2 = p3
        p2 = (x2, y1)
        p4 = (x1, y2)

        tops = []
        p_mid = self.cude_center_of_two_points(p1, p3)
        p_1_mid = self.cude_center_of_two_points(p1, p_mid)
        p_2_mid = self.cude_center_of_two_points(p2, p_mid)
        tops.append(self.fourth_center_of_later(p_mid, p_1_mid))
        tops.append(self.fourth_center_of_later(p_mid, p_2_mid))

        bots = []
        p14 = self.cude_center_of_two_points(p1, p4)
        p34 = self.cude_center_of_two_points(p3, p4)
        left_bots = self.four_cude_center_from_two_points(p14, p34, left=True)
        right_bots = self.four_cude_center_from_two_points(p_mid, p3, left=False)
        bots.extend(left_bots)
        bots.extend(right_bots)

        all_points = []
        all_points.extend(tops)
        all_points.extend(bots)
        return all_points

    def twelve_cores(self, p1: tuple, p3: tuple):
        assert len(p1) == 2 and len(p3) == 2, "p1 and p3 must be a tuple of length 2"
        x1, y1 = p1
        x2, y2 = p3
        p2 = (x2, y1)
        p4 = (x1, y2)
        p_mid = self.cude_center_of_two_points(p1, p3)

        tops = []
        p12 = self.cude_center_of_two_points(p1, p2)
        p_1_mid = self.cude_center_of_two_points(p1, p_mid)
        p_12_1_mid = self.cude_center_of_two_points(p12, p_1_mid)
        tops.append(p_12_1_mid)
        p14 = self.cude_center_of_two_points(p1, p4)
        tops.append(self.fourth_center_of_later(p14, p_1_mid))

        p_2_mid = self.cude_center_of_two_points(p2, p_mid)
        p_12_2_mid = self.cude_center_of_two_points(p12, p_2_mid)
        tops.append(p_12_2_mid)
        p23 = self.cude_center_of_two_points(p2, p3)
        tops.append(self.fourth_center_of_later(p23, p_2_mid))

        bots = []
        p34 = self.cude_center_of_two_points(p3, p4)
        left_bots = self.four_cude_center_from_two_points(p14, p34, left=True)
        right_bots = self.four_cude_center_from_two_points(p_mid, p3, left=False)
        bots.extend(left_bots)
        bots.extend(right_bots)

        all_points = []
        all_points.extend(tops)
        all_points.extend(bots)
        return all_points


class BiopsyPlanWithBoundary:
    def __init__(self, prostate_file: str, specimen_file: str, result_path: str):
        self.prostate_file = Path(prostate_file)
        self.specimen_file = Path(specimen_file)
        self.result_path = Path(result_path)

    def plan(self, plan_method: CorePlanType = CorePlanType.TWELVE):
        colors = vtk.vtkNamedColors()

        prostate_reader = vtk.vtkSTLReader()
        prostate_reader.SetFileName(str(self.prostate_file))
        prostate_reader.Update()

        center_plane = vtk.vtkPlane()
        center_plane.SetOrigin(prostate_reader.GetOutput().GetCenter())
        center_plane.SetNormal(0, 0, 1)

        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(prostate_reader.GetOutputPort())
        cutter.SetCutFunction(center_plane)
        cutter.GenerateValues(1, 0, 0)

        self.print_polydata_info(cutter.GetOutput())

        mapper_poly = vtk.vtkPolyDataMapper()
        mapper_poly.SetInputData(prostate_reader.GetOutput())
        actor_poly = vtk.vtkActor()
        actor_poly.SetMapper(mapper_poly)
        actor_poly.GetProperty().SetColor(colors.GetColor3d("Red"))
        actor_poly.GetProperty().SetOpacity(0.5)

        mapper_cutter = vtk.vtkPolyDataMapper()
        mapper_cutter.SetInputConnection(cutter.GetOutputPort())
        actor_cutter = vtk.vtkActor()
        actor_cutter.SetMapper(mapper_cutter)
        actor_cutter.GetProperty().SetColor(colors.GetColor3d("Green"))

        # if plan_method == CorePlanType.TEN:
        # core_points = self.ten_cores()
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor_poly)
        renderer.AddActor(actor_cutter)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        window.SetSize(800, 800)
        window.Render()

        window_to_img = vtk.vtkWindowToImageFilter()
        window_to_img.SetInput(window)
        window_to_img.SetInputBufferTypeToRGB()
        window_to_img.ReadFrontBufferOff()
        window_to_img.Update()

        prefix = "ten_core" if plan_method == CorePlanType.TEN else "twelve_core"
        png_file_name = self.result_path.joinpath(
            f"{prefix}_{self.prostate_file.stem}.png"
        )
        png_writer = vtk.vtkPNGWriter()
        png_writer.SetFileName(png_file_name)
        png_writer.SetInputConnection(window_to_img.GetOutputPort())
        png_writer.Write()

        interactor.Initialize()
        interactor.Start()

    @staticmethod
    def print_polydata_info(polydata: vtk.vtkPolyData):
        print(f"Number of points: {polydata.GetNumberOfPoints()}")
        print(f"Number of lines: {polydata.GetNumberOfLines()}")
        print(f"Number of polygons: {polydata.GetNumberOfPolys()}")
        print(f"Number of cells: {polydata.GetNumberOfCells()}")
        print(f"Number of vertices: {polydata.GetNumberOfVerts()}")
        print(f"Number of strips: {polydata.GetNumberOfStrips()}")
        print(f"Number of pieces: {polydata.GetNumberOfPieces()}")

    def ten_cores():
        pass
