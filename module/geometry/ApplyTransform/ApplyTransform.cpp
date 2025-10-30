#include <vistle/core/object.h>
#include <vistle/core/points.h>
#include <vistle/alg/objalg.h>

#include "ApplyTransform.h"

#ifdef APPLYTRANSFORMVTKM
#include <viskores/cont/DataSet.h>
#include <viskores/filter/field_transform/PointTransform.h>
#include <vistle/vtkm/convert.h>
#endif

MODULE_MAIN(ApplyTransform)

using namespace vistle;

ApplyTransform::ApplyTransform(const std::string &name, int moduleID, mpi::communicator comm)
: Module(name, moduleID, comm)
{
    createInputPort("grid_in", "grid or geometry");
    createOutputPort("grid_out", "unconnected points");

    addResultCache(m_gridCache);
    addResultCache(m_resultCache);
}

ApplyTransform::~ApplyTransform()
{}

// -----------------------------
// CPU helper
// -----------------------------
vistle::Coords::ptr ApplyTransform::applyTransformCpu(const vistle::Coords &coords)
{
    auto T = coords.getTransform();
    auto clone = coords.clone();

    if (!T.isIdentity()) {
        clone->resetCoords();
        clone->setTransform(vistle::Matrix4::Identity());
        vistle::Index ncoords = coords.getSize();
        clone->setSize(ncoords);

        auto *x = coords.x().data();
        auto *y = coords.y().data();
        auto *z = coords.z().data();

        auto *nx = clone->x().data();
        auto *ny = clone->y().data();
        auto *nz = clone->z().data();

        for (vistle::Index i = 0; i < ncoords; ++i) {
            auto p = T * vistle::Vector4(x[i], y[i], z[i], vistle::Scalar(1));
            nx[i] = p[0] / p[3];
            ny[i] = p[1] / p[3];
            nz[i] = p[2] / p[3];
        }
    }

    return clone;
}

bool ApplyTransform::compute()
{
    auto o = expect<Object>("grid_in");
    auto split = splitContainerObject(o);
    auto coords = Coords::as(split.geometry);
    if (!coords) {
        sendError("no coordinates on input");
        return true;
    }

    Object::ptr out;
    if (auto resultEntry = m_resultCache.getOrLock(o->getName(), out)) {

        Object::ptr outGrid;
        if (auto gridEntry = m_gridCache.getOrLock(coords->getName(), outGrid)) {

            auto T = coords->getTransform();

            if (T.isIdentity()) {
                // nothing to do, just clone
                auto clone = coords->clone();
                updateMeta(clone);
                outGrid = clone;
            } else {

#ifdef APPLYTRANSFORMVTKM
                // =====================================================
                // GPU / Viskores path
                // =====================================================
                viskores::cont::DataSet inDs;
                vistle::vtkmSetGrid(inDs, coords);

                // Set up the point transform filter
                viskores::filter::field_transform::PointTransform filter;

                // Copy Vistle Matrix4 into Viskores 4x4
                viskores::Matrix<vistle::Scalar, 4, 4> M;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        M[r][c] = T(r, c);
                filter.SetTransform(M);

                // Run on GPU
                (void)filter.Execute(inDs);

                auto clone = coords->clone();
                clone->setTransform(vistle::Matrix4::Identity());
                updateMeta(clone);
                outGrid = clone;
#else
                // =====================================================
                // CPU path
                // =====================================================
                auto clone = applyTransformCpu(*coords);
                updateMeta(clone);
                outGrid = clone;
#endif
            }

            m_gridCache.storeAndUnlock(gridEntry, outGrid);
        }

        // reattach mapped data 
        if (split.mapped) {
            auto mapping = split.mapped->guessMapping();
            if (mapping != DataBase::Vertex) {
                sendError("data has to be mapped per vertex");
                out = outGrid;
            } else {
                auto data = split.mapped->clone();
                data->setGrid(outGrid);
                updateMeta(data);
                out = data;
            }
        } else {
            out = outGrid;
        }

        m_resultCache.storeAndUnlock(resultEntry, out);
    }

    addObject("grid_out", out);
    return true;
}
